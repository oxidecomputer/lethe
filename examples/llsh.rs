use rustyline::error::ReadlineError;

fn main() -> Result<(), Box<dyn std::error::Error>> {
    let mut rl = rustyline::Editor::<()>::new()?;

    loop {
        match rl.readline(">> ") {
            Ok(line) => println!("{line}"),
            Err(ReadlineError::Eof) => {
                println!("exiting.");
                break;
            }
            Err(ReadlineError::Interrupted) => {
                println!("^C");
            }
            Err(ReadlineError::WindowResized) => {
                println!("wheeeee");
            }
            Err(ReadlineError::Io(e)) => return Err(e.into()),
            Err(e) => {
                println!("unexpected error: {e:?}");
                break;
            }
        }
    }

    Ok(())
}
