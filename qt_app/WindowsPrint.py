import subprocess
import time

for i in range(3):
    subprocess.run(
        ["powershell", "-Command", "$input | Out-Printer -Name 'Dispenser'"],
        input="\f",
        text=True,
        capture_output=True
    )
    time.sleep(0.5)