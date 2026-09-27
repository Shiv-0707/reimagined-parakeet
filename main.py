from future import annotations

import argparse
import logging
from dataclasses import dataclass

LOGGER = logging.getLogger(name)

@dataclass(frozen=True)
class Greeting:
  """Immutable value object representing a greeting."""

recipient: str

def render(self) -> str:
  """Return the formatted greeting message."""
  return f"Hello, {self.recipient}!"

def parse_args(argv: list[str] | None = None) -> argparse.Namespace:
  """Parse command-line arguments."""
  parser = argparse.ArgumentParser(
  description="Reimagined Parakeet - a clean Python starter project."
  )
  parser.add_argument(
  "--name",
  default="World",
  help="Name to greet (default: World).",
  )
  parser.add_argument(
  "--verbose",
  action="store_true",
  help="Enable debug logging.",
  )
  return parser.parse_args(argv)

def configure_logging(verbose: bool) -> None:
  """Configure application logging."""
  logging.basicConfig(
  level=logging.DEBUG if verbose else logging.INFO,
  format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
  )

def main(argv: list[str] | None = None) -> int:
  """Application entry point."""
  args = parse_args(argv)
  configure_logging(args.verbose)

greeting = Greeting(recipient=args.name)
message = greeting.render()
LOGGER.info(message)
print(message)
return 0

if name == "main":
  raise SystemExit(main())
  
