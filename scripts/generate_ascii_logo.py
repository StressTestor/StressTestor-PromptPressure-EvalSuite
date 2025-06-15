"""
Generate an ASCII art logo for PromptPressure Eval Suite.
"""

def create_ascii_logo():
    logo = """
  ____                      ____               _           
 |  _ \ _ __ _ __ _ __ ___ |  _ \ _ __ ___  __| | ___ _ __ 
 | |_) | '__| '__| '_ ` _ \| |_) | '__/ _ \/ _` |/ _ \ '__|
 |  __/| |  | |  | | | | | |  __/| | |  __/ (_| |  __/ |   
 |_|   |_|  |_|  |_| |_| |_|_|   |_|  \___|\__,_|\___|_|   
                                                          
  _____           _   _               _____     _     _   
 |  ___|_ _ _ __| |_(_)_ __   __ _  | ____|___| |__ | |_ 
 | |_ / _` | '__| __| | '_ \ / _` | |  _| / __| '_ \| __|
 |  _| (_| | |  | |_| | | | | (_| | | |___\__ \ | | | |_ 
 |_|  \__,_|_|   \__|_|_| |_|\__, | |_____|___/_| |_|\__|
                             |___/                         
    """
    return logo

if __name__ == "__main__":
    logo = create_ascii_logo()
    print(logo)
    
    # Save to file
    with open("../assets/logo.txt", "w") as f:
        f.write(logo)
    print("Logo saved to assets/logo.txt")
