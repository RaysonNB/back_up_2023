import subprocess

text = "你好，我係"
subprocess.call(['espeak-ng', '-v', 'yue', text])
