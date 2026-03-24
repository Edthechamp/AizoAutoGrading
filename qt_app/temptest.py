import json
from checkAns import GradeTest

# The dictionary from your image
answers = {
    'code': '123456', 
    'a': {'1': 'A', '2': 'B', '3': 'C', '4': 'D', '5': 'C', '6': 'B', '7': 'A', '8': 'B', '9': 'C', '10': 'D'}, 
    'b': {'1': 'B', '2': 'B', '3': 'B', '4': 'D', '5': 'A', '6': 'C', '7': 'B', '8': 'A', '9': 'D', '10': 'C'}, 
    'c': {'1': 'A', '2': 'C', '3': 'B', '4': 'A', '5': 'B', '6': 'B', '7': 'D', '8': 'C', '9': 'D', '10': 'A'}
}

# Conversion logic
test = {
    "StudentID": answers.get("code", ""),
    "answers": {key: value for key, value in answers.items() if key != "code"}
}

# Print the formatted JSON output
print(test)
print(GradeTest(test))
