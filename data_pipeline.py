
if __name__ == '__main__':
    import subprocess

    subprocess.run("python create_database.py & python create_domains.py", shell=True)