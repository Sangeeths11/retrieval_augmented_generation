import re
from typing import List, Dict, Any

def clean_text(text: str) -> str:
    """
    Clean text by removing extra whitespace, normalizing newlines, etc.
    
    Args:
        text: Text to clean
        
    Returns:
        Cleaned text
    """

    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r' +', ' ', text)
    text = text.strip()
    
    return text

def extract_metadata(text: str) -> Dict[str, Any]:
    """
    Extract metadata from text like title, authors, etc.
    
    Args:
        text: Text to extract metadata from
        
    Returns:
        Dictionary of metadata
    """
    metadata = {}
    
    # TODO: Make this more accurate and stufff... with vision
    lines = [line.strip() for line in text.split('\n') if line.strip()]
    if lines:
        metadata['title'] = lines[0]
    
    # TODO: Make this more accurate and stufff...
    author_lines = [line for line in lines if re.search(r'(?i)author|^by\b', line)]
    if author_lines:
        metadata['authors'] = author_lines[0]
    
    return metadata

def split_by_section(text: str) -> List[Dict[str, str]]:
    """
    Split text into sections based on headings.
    
    Args:
        text: Text to split
        
    Returns:
        List of dictionaries with section title and content
    """
    # Simple heuristic for the first we can adjust this to be more accurate and stufff...
    section_pattern = r'(?:\n|^)([A-Z][A-Za-z0-9 ]{1,50}[:.?!]?)(?:\n)+'
    
    matches = list(re.finditer(section_pattern, text))
    sections = []
    
    for i, match in enumerate(matches):
        title = match.group(1).strip()
        start = match.end()
        
        # Find end of section
        if i < len(matches) - 1:
            end = matches[i + 1].start()
        else:
            end = len(text)
        
        content = text[start:end].strip()
        sections.append({"title": title, "content": content})
    
    # If no sections found, return the whole text as one section
    if not sections:
        sections.append({"title": "Document", "content": text})
    
    return sections 