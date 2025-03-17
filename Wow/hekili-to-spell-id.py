import re
from typing import List


def extract_ids(text: str) -> List[int]:
    """Extract all numeric IDs following 'id =' patterns from text.

    Args:
        text: Input text containing id assignments

    Returns:
        List of integer IDs found in the text
    """
    pattern = r'id\s*=\s*(\d+)'
    matches = re.findall(pattern, text)
    return [int(m) for m in matches]


# Usage example:
if __name__ == "__main__":
    sample_text = """



































    """
    print(extract_ids(sample_text))  # [22812, 5487, 106951]
