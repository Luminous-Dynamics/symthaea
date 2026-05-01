"""Small CLI helpers shared by research scripts."""


def print_banner(title: str, width: int = 76) -> None:
    line = "=" * width
    print(line)
    print(title.center(width))
    print(line)
    print()


def print_section(title: str, width: int = 76, fill: str = "-") -> None:
    line = fill * width
    print(line)
    print(f" {title}")
    print(line)
