import subprocess
import re
from pathlib import Path
import sqlite3
import time


#TODO: create database table where to save student asnwers
conn = sqlite3.connect("AAG.db")
cursor = conn.cursor()



cursor.execute('''
    CREATE TABLE IF NOT EXISTS students (
        student_id TEXT UNIQUE,
        score TEXT
    )
''')


def dispensePage(timeout=25):

    result = subprocess.run(
        ["lp", "-d", "Dispenser"], 
        input="\f", 
        text=True, 
        capture_output=True
    )

    if result.returncode != 0:
        print(f"Genuine hz kam janotiek lai butu sads error")
        return False
    
    match = re.search(r'request id is (\S+)', result.stdout)
    if not match:
        print(f"Failed to get job ID from output: {result.stdout}")
        return False
        
    job_id = match.group(1)

    start_time = time.time()
    while time.time() - start_time < timeout:

        completed = subprocess.run(["lpstat", "-W", "completed"], capture_output=True, text=True)
        
        if job_id in completed.stdout:
            print("Dispensed successfully.")
            return True
            
        time.sleep(0.5)
    
    return False

def getFilePath(file):
    base_path = Path(__file__).parent.parent
    return base_path / "Resources" / file


def saveAnswers(grades):
    data = (grades["code"],grades["scores"])
    cursor.execute('''
    INSERT INTO students (student_id, score) 
    VALUES (?, ?)
''', data)
    conn.commit()
