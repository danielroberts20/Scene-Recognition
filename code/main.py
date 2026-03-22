import run1
import run2
import run3
from scenes import Scene


def main():
    print("\nScenes:")
    for index, scene in enumerate(Scene):
       print(f"\t({index}) | {scene.title} ({scene.index})")
    separator()

    run1.main()
    separator()

    run2.main()
    separator()

    run3.main()

def separator(num: int = 20):
    line = "-" * num
    print(f"\n{line}\n")

if __name__ == "__main__":
    main()
