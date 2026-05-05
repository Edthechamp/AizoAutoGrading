import os
import webbrowser

def display_score_page(score):
    
    display_score = str(score)
    
    html_content = f"""
    <!DOCTYPE html>
    <html lang="en">
    <head>
        <meta charset="UTF-8">
        <meta name="viewport" content="width=device-width, initial-scale=1.0">
        <title>Your Results</title>
        <style>
            body {{
                font-family: 'Segoe UI', Tahoma, Geneva, Verdana, sans-serif;
                background: linear-gradient(135deg, #667eea 0%, #764ba2 100%);
                height: 100vh;
                margin: 0;
                display: flex;
                justify-content: center;
                align-items: center;
                color: #2d3436;
            }}
            .card {{
                background: white;
                padding: 3rem;
                border-radius: 20px;
                box-shadow: 0 15px 35px rgba(0,0,0,0.2);
                text-align: center;
                max-width: 400px;
                width: 90%;
            }}
            h1 {{
                margin-top: 0;
                color: #6c5ce7;
                font-size: 1.5rem;
                text-transform: uppercase;
                letter-spacing: 2px;
            }}
            .score-container {{
                font-size: 3rem;
                font-weight: bold;
                margin: 1.5rem 0;
            }}
            .score-highlight {{
                color: #a29bfe;
            }}
            .out-of {{
                font-size: 1.2rem;
                color: #636e72;
            }}
            .footer-text {{
                color: #b2bec3;
                font-size: 0.9rem;
            }}
        </style>
    </head>
    <body>
        <div class="card">
            <h1>Quiz Results</h1>
            <div class="score-container">
                You scored:<br>
                <span class="score-highlight">{display_score}</span>
                <span class="out-of">out of 5 points</span>
            </div>
        </div>
    </body>
    </html>
    """

    
    
    with open("results.html", "w") as f:
        f.write(html_content)

    path = os.path.abspath("results.html")
    file_url = f"file:///{path}".replace("\\", "/")

    try:
        browser = webbrowser.get('google-chrome')
        browser.open(file_url)
    except webbrowser.Error:
        webbrowser.open(file_url)

display_score_page(4)