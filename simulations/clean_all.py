import os
import re
import manifest

def clean_analyzers():
    print(manifest.CURRENT_DIR)
    for file in os.listdir(manifest.CURRENT_DIR):
        if re.match("run_analyzer_.*sh", file) or re.match("wait_analyzer_.*sh", file):
            os.remove(os.path.join(manifest.CURRENT_DIR,file))

def clean_logs():
    log_dir= manifest.CURRENT_DIR / "log"
    for file in os.listdir(log_dir):
            os.remove(os.path.join(log_dir,file))

if __name__ == "__main__":
    clean_analyzers()
    clean_logs()