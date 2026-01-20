

def load_text(path, by_lines=False):
    with open(path, "r") as fp:
        if by_lines:
            return fp.readlines()
        else:
            return fp.read()


def load_prompt(role, name):
    return load_text(f"./packages/prompts/{role}/{name}.txt")
