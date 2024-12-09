import os
import sys
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.join(current_dir, '..')  # Going up one level to the root directory
app_dir = os.path.join(root_dir, 'scripts')
sys.path.insert(0, app_dir)  
from preprocess import preprocess_text

def test_preprocess_text():
    assert preprocess_text("Hello World!") == "hello world"
    assert preprocess_text("Special   chars!@#") == "special chars"
    print("Preprocessing tests passed.")

if __name__ == "__main__":
    test_preprocess_text()
