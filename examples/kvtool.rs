// This Source Code Form is subject to the terms of the Mozilla Public
// License, v. 2.0. If a copy of the MPL was not distributed with this
// file, You can obtain one at https://mozilla.org/MPL/2.0/.

use clap::Parser;
use anyhow::{Context, anyhow, bail};
use std::io::{Seek, Read, Write, SeekFrom};
use std::cell::RefCell;
use num_traits::FromPrimitive;
use lethe::low_level::Flash;

#[derive(Parser)]
struct Kvtool {
    #[clap(short)]
    sector_size: u8,

    #[clap(subcommand)]
    cmd: Cmd,

    image_file: std::path::PathBuf,
}

#[derive(Parser)]
enum Cmd {
    Check,
    Repair,
    Format {
        #[clap(arg_enum)]
        space: ArgSpace,
    },
    Erase {
        #[clap(arg_enum)]
        space: ArgSpace,
        #[clap(subcommand)]
        target: EraseTarget,
    },
    Write {
        key: String,
        value: String,
    },
    Read {
        key: String,
    },
    Locate {
        key: String,
    },
    Dump {
        #[clap(arg_enum)]
        space: ArgSpace,
    },
    Evacuate {
        #[clap(arg_enum)]
        space: ArgSpace,
    },
    Corrupt {
        #[clap(arg_enum)]
        space: ArgSpace,
        #[clap(subcommand)]
        target: EraseTarget,
    },
}

#[derive(Parser)]
enum EraseTarget {
    All,
    Sector {
        number: u32,
    },
}

fn main() -> Result<(), anyhow::Error> {
    let args = Kvtool::from_args();

    match args.sector_size {
        32 => specialized_main::<32>(args)?,
        x => bail!("unsupported sector size {x}"),
    }

    Ok(())
}

#[derive(Copy, Clone, Debug, clap::ArgEnum)]
enum ArgSpace {
    Zero,
    One,
}

impl From<ArgSpace> for Space {
    fn from(a: ArgSpace) -> Self {
        match a {
            ArgSpace::Zero => Self::Zero,
            ArgSpace::One => Self::One,
        }
    }
}

fn specialized_main<const S: usize>(args: Kvtool) -> Result<(), anyhow::Error> {
    let mut img = FlashImage::<S>::open(&args.image_file)
        .with_context(|| {
            format!("opening image file {}", args.image_file.display())
        })?;

    let mut buffer0 = [0; S];
    let mut buffer1 = [0; S];

    match args.cmd {
        Cmd::Repair => {
            with_mounted_image(img, |mut store| {
                match store.repair() {
                    Ok(()) => println!("success"),
                    Err(e) => println!("error: {e:?}"),
                }
                Ok(())
            })?;
        }
        Cmd::Check => {
            for space in Space::ALL {
                println!("checking space {:?}", space);

                let result = lethe::low_level::check(
                    &mut img,
                    &mut buffer0,
                    space,
                )?;

                use lethe::low_level::CheckResult;
                match result {
                    CheckResult::ValidLog { generation, end, tail_erased, incomplete_write } => {
                        println!("- contains valid log of {end} sectors");
                        println!("- marked as generation {generation}");
                        println!("- tail of space {} erased",
                            if tail_erased { "is" } else { "IS NOT" },
                        );
                        println!("- final entry in log is {}",
                            if incomplete_write { "INCOMPLETE" } else { "complete" },
                        );
                    }
                    CheckResult::Bad(e) => {
                        println!("- could not validate: {e:?}");
                    }
                }
            }
        }
        Cmd::Erase { space, target } => {
            let space = Space::from(space);
            match target {
                EraseTarget::All => {
                    println!("erasing space {space:?}");
                    img.erase_space(space)?;
                }
                EraseTarget::Sector { number } => {
                    println!("erasing sector {number} in space {space:?}");
                    img.erase_sector(space, number)?;
                }
            }
        }
        Cmd::Corrupt { space, target } => {
            let space = Space::from(space);
            match target {
                EraseTarget::All => {
                    println!("corrupting space {space:?}");
                    img.erase_space(space)?;
                    let nonsense = [0xAA; S];
                    for s in 0..img.sectors_per_space() {
                        img.program_sector(space, s, &nonsense)?;
                    }
                }
                EraseTarget::Sector { number } => {
                    println!("corrupting sector {number} in space {space:?}");
                    img.erase_sector(space, number)?;
                    let nonsense = [0xAA; S];
                    img.program_sector(space, number, &nonsense)?;
                }
            }
        }
        Cmd::Format { space } => {
            println!("formatting space {space:?}");
            let r = lethe::low_level::format(
                &mut img,
                &mut buffer0,
                space.into(),
                0,
            );
            use lethe::low_level::FormatError;
            match r {
                Ok(()) => println!("success"),
                Err(FormatError::NeedsErase) => {
                    println!("space must be erased first");
                }
                Err(FormatError::Flash(e)) => {
                    return Err(e).context("accessing image");
                }
            }
        }
        Cmd::Write { key, value } => {
            with_writable_mounted_image(img, |mut store| {
                use lethe::low_level::WriteError;
                match store.write_kv(key.as_bytes(), value.as_bytes()) {
                    Ok(()) => println!("ok"),
                    Err(WriteError::NoSpace) => println!("no space"),
                    Err(e) => println!("error: {e:?}"),
                }
                Ok(())
            })?;
        }
        Cmd::Locate { key } => {
            with_mounted_image(img, |store| {
                match store.locate_kv(key.as_bytes()) {
                    Ok(Some(x)) => println!("found at sector {x}"),
                    Ok(None) => println!("not found"),
                    Err(e) => println!("error: {e:?}"),
                }
                Ok(())
            })?;
        }
        Cmd::Read { key } => {
            with_mounted_image(img, |store| {
                let mut out = [0; 1024];
                match store.read_kv(key.as_bytes(), &mut out) {
                    Ok(Some(n)) => {
                        println!("{}", pretty_hex::pretty_hex(&&out[..n]));
                    }
                    Ok(None) => println!("not found"),
                    Err(e) => println!("error: {e:?}"),
                }
                Ok(())
            })?;
        }
        Cmd::Dump { space } => {
            let space = Space::from(space);
            let result = lethe::low_level::check(
                &mut img,
                &mut buffer0,
                space,
            )?;

            use lethe::low_level::CheckResult;
            let end = match result {
                CheckResult::ValidLog { end, .. } => {
                    end
                }
                CheckResult::Bad(e) => {
                    bail!("- could not validate: {e:?}");
                }
            };

            println!("dumping log contents from space {space:?}");

            let mut seen_keys = std::collections::HashMap::new();
            let mut sector = lethe::low_level::Constants::<FlashImage<S>>::HEADER_SECTORS;

            while sector < end {
                println!("entry at sector {sector}");
                let entry = lethe::low_level::read_entry_from_head(
                    &mut img,
                    &mut buffer0,
                    space,
                    sector,
                ).map_err(|e| {
                    anyhow!("failed to read: {e:?}")
                })?;
                let kst = lethe::low_level::KnownSubtypes::from_u8(entry.meta.subtype);
                let next_sector = entry.next_sector;
                let contents_length = entry.meta.contents_length.get();

                println!("- content length {}", contents_length);
                println!("- subtype {:#02x} ({})",
                    entry.meta.subtype,
                    match kst {
                        Some(t) => format!("{:?}", t),
                        None => "unknown".to_string(),
                    });

                match kst {
                    Some(lethe::low_level::KnownSubtypes::Data) | Some(lethe::low_level::KnownSubtypes::Delete) => {
                        let (subheader, _) = lethe::low_level::cast_prefix::<lethe::low_level::DataSubMeta>(entry.submeta);
                        println!("- key hash {:#08x}", subheader.key_hash.get());

                        let mut key = vec![0; subheader.key_length.get() as usize];
                        lethe::low_level::read_contents(
                            &mut img,
                            &mut buffer0,
                            space,
                            sector,
                            0,
                            &mut key,
                        ).map_err(|e| {
                            anyhow!("failed to read key: {e:?}")
                        })?;

                        if let Some(prev) = seen_keys.insert(key.clone(), sector) {
                            println!("- supercedes entry at {prev}");
                        }
                        println!("Key {}", pretty_hex::pretty_hex(&key));

                        if kst == Some(lethe::low_level::KnownSubtypes::Delete) {
                            println!("- this entry deletes the key");
                        } else {
                            let value_len = contents_length - key.len() as u32;
                            let mut val = vec![0; value_len as usize];
                            lethe::low_level::read_contents(
                                &mut img,
                                &mut buffer0,
                                space,
                                sector,
                                key.len() as u32,
                                &mut val,
                            ).map_err(|e| {
                                anyhow!("failed to read val: {e:?}")
                            })?;
                            println!("Value {}", pretty_hex::pretty_hex(&val));
                        }

                    }
                    _ => (),
                }

                println!();

                sector = next_sector;
            }

            
        }
        Cmd::Evacuate { space } => {
            let space = Space::from(space);
            let result = lethe::low_level::check(
                &mut img,
                &mut buffer0,
                space,
            )?;

            use lethe::low_level::CheckResult;
            let end = match result {
                CheckResult::ValidLog { end, .. } => {
                    end
                }
                CheckResult::Bad(e) => {
                    bail!("- could not validate: {e:?}");
                }
            };

            println!("evacuating contents of space {:?} => {:?}",
                space, space.other());

            lethe::low_level::evacuate(
                &mut img,
                &mut buffer0,
                &mut buffer1,
                space,
                end,
            ).map_err(|e| {
                anyhow!("flash access error: {e:?}")
            })?;

            println!("done");
        }
    }

    Ok(())
}

fn with_mounted_image<const S: usize>(
    img: FlashImage<S>,
    body: impl FnOnce(lethe::Store<'_, FlashImage<S>>) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    let mut buffers = lethe::StoreBuffers {
        b0: [0; S],
        b1: [0; S],
    };
    let store = match lethe::mount(img, &mut buffers) {
        Err(e) => bail!("could not mount: {:?}", e.cause()),
        Ok(store) => store,
    };

    body(store)
}

fn with_writable_mounted_image<const S: usize>(
    img: FlashImage<S>,
    body: impl FnOnce(lethe::WritableStore<'_, FlashImage<S>>) -> anyhow::Result<()>,
) -> anyhow::Result<()> {
    with_mounted_image(img, |store| {
        if !store.can_mount_writable() {
            bail!("can't mount store writable due to errors");
        }

        let store = store.mount_writable().map_err(|_| ()).unwrap();
        body(store)
    })
}

struct FlashImage<const S: usize> {
    file: RefCell<std::fs::File>,
    sectors_per_space: u32,
}

impl<const S: usize> FlashImage<S> {
    fn open(path: impl AsRef<std::path::Path>) -> Result<Self, anyhow::Error> {
        let file = std::fs::OpenOptions::new()
            .read(true)
            .write(true)
            .create(false)
            .open(path)?;
        let metadata = file.metadata()?;
        let file_len = metadata.len();

        if file_len % S as u64 != 0 {
            bail!("file is not a whole number of sectors in length");
        }

        let sectors = file_len / S as u64;
        if sectors % 2 != 0 {
            bail!("file contains an odd number of sectors");
        }

        let sectors_per_space = u32::try_from(sectors / 2)
            .context("file too large")?;

        Ok(Self {
            file: file.into(),
            sectors_per_space,
        })
    }

    fn erase_sector(&mut self, space: Space, sector: u32) -> Result<(), anyhow::Error> {
        let global_sector = usize::from(space) as u64 * self.sectors_per_space as u64 + sector as u64;
        let global_offset = global_sector * S as u64;
        let mut file = self.file.borrow_mut();
        file.seek(SeekFrom::Start(global_offset))?;
        let data = [0xFF; S];
        file.write_all(&data)?;
        Ok(())
    }
}

use lethe::low_level::Space;

impl<const S: usize> lethe::low_level::Flash for FlashImage<S> {
    type Sector = [u8; S];
    type Error = std::io::Error;

    fn sectors_per_space(&self) -> u32 {
        self.sectors_per_space
    }

    fn read_sector(&self, space: Space, index: u32, dest: &mut Self::Sector) -> Result<(), Self::Error> {
        let global_sector = usize::from(space) as u64 * self.sectors_per_space as u64 + index as u64;
        let global_offset = global_sector * S as u64;

        let mut file = self.file.borrow_mut();

        file.seek(SeekFrom::Start(global_offset))?;
        file.read_exact(dest)?;

        /*
        println!("sector {index}");
        println!("{}", pretty_hex::pretty_hex(dest));
        */
        Ok(())
    }

    fn can_program_sector(&self, space: Space, index: u32) -> Result<bool, Self::Error> {
        let mut b = [0; S];
        self.read_sector(space, index, &mut b)?;
        Ok(b.iter().all(|&byte| byte == 0xFF))
    }

    fn can_read_sector(&self, _space: Space, _index: u32) -> Result<bool, Self::Error> {
        // This backend allows arbitrary reads.
        Ok(true)
    }

    fn program_sector(&mut self, space: Space, index: u32, data: &Self::Sector) -> Result<(), Self::Error> {
        let global_sector = usize::from(space) as u64 * self.sectors_per_space as u64 + index as u64;
        let global_offset = global_sector * S as u64;

        let mut file = self.file.borrow_mut();

        file.seek(SeekFrom::Start(global_offset))?;
        file.write_all(data)?;
        Ok(())
    }

    fn erase_space(&mut self, space: Space) -> Result<(), Self::Error> {
        let global_sector = usize::from(space) as u64 * self.sectors_per_space as u64;
        let global_offset = global_sector * S as u64;

        let mut file = self.file.borrow_mut();

        file.seek(SeekFrom::Start(global_offset))?;
        let data = [0xFF; S];
        for _ in 0..self.sectors_per_space {
            file.write_all(&data)?;
        }
        Ok(())
    }
}
