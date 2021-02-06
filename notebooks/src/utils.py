import os


def print_params(p, bds):
    s = ""
    for p, b in zip(p, bds):
        tmp = f"| {b['name']}: {p:.2f} "
        s += tmp
    print(s + "|" + "\n")


def call_counter(func):
    """Call counter decorator"""

    def helper(*args, **kwargs):
        helper.calls += 1
        return func(*args, **kwargs)

    helper.calls = 0
    helper.__name__ = func.__name__
    return helper


def setup_folders(f="models/"):
    if not os.path.exists(f):
        os.makedirs(f)
        print(f"Directory: {f} created")


def clean_folder(folder):
    print(f"Deleting models from: {folder}")
    for the_file in os.listdir(folder):
        file_path = os.path.join(folder, the_file)
        try:
            if os.path.isfile(file_path):
                os.unlink(file_path)
        except Exception as e:
            print(e)
