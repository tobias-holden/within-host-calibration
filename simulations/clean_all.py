import os
import re
import manifest

def clean_analyzers():
    for file in os.listdir(manifest.CURRENT_DIR):
        if re.match("run_analyzer_.*sh", file) or re.match("wait_analyzer_.*sh", file):
            os.remove(file)

if __name__ == "__main__":
    clean_analyzers()