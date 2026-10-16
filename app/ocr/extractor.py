import re
from typing import Optional, Dict, List, Any
from app.utils.helpers import logger

# Alias definitions ordered by descending length to match longest first
BIOMARKER_ALIASES = {
    "hemoglobin": ["hemoglobin", "hgb", "hb"],
    "rbc": ["total rbc count", "rbc count", "rbc"],
    "wbc": ["white blood cell count", "total wbc count", "wbc count", "wbc"],
    "platelets": ["platelet count", "platelets", "plt"]
}

def clean_line(line: str) -> str:
    """Normalize a line: lowercase, replace multiple spaces, and strip junk characters."""
    line = line.lower()
    line = re.sub(r'\s+', ' ', line)
    # Strip common leading/trailing layout characters or OCR artifacts
    line = line.strip(" _-=*|:,.()")
    return line

def merge_broken_lines(lines: List[str]) -> List[str]:
    """
    Merge lines where a biomarker is declared on one line (without numeric results),
    and the numeric results appear on the immediate next line.
    """
    merged_lines = []
    i = 0
    n = len(lines)
    while i < n:
        line = lines[i].strip()
        if not line:
            i += 1
            continue
        
        # Check if this line contains any biomarker alias
        has_biomarker = False
        for alias_list in BIOMARKER_ALIASES.values():
            for alias in alias_list:
                if re.search(r'\b' + re.escape(alias) + r'\b', line):
                    has_biomarker = True
                    break
            if has_biomarker:
                break
        
        # Check if this line contains any digits
        has_digits = any(c.isdigit() for c in line)
        
        # If line lists a biomarker but has no digits, check if the next line has digits to merge
        if has_biomarker and not has_digits and (i + 1 < n):
            next_line = lines[i+1].strip()
            # If the next line contains digits, merge them
            if any(c.isdigit() for c in next_line):
                logger.info(f"Merging broken lines: '{line}' + '{next_line}'")
                line = line + " " + next_line
                i += 2  # Skip next line as it was merged
            else:
                i += 1
        else:
            i += 1
            
        merged_lines.append(line)
        
    return merged_lines

def clean_numeric_string(s: str) -> str:
    """Replace common OCR character misreads in numeric strings."""
    # o / O -> 0
    s = s.replace('o', '0').replace('O', '0')
    # l / I / | -> 1
    s = s.replace('l', '1').replace('I', '1').replace('|', '1')
    # Strip any other character except digits and decimal point
    s = re.sub(r'[^0-9.]', '', s)
    return s

def try_parse_numeric(token: str) -> Optional[str]:
    """
    Evaluates if a string token is a numeric candidate.
    Cleans surrounding punctuation and handles common units/labels.
    """
    cleaned = token.strip("(),;[]{}*:/\\-")
    if not cleaned:
        return None
        
    # Ignore common text fields/units/labels
    ignored_words = {"low", "high", "normal", "nil", "negative", "cumm", "g/dl", "mill/cumm", "borderline", "result", "value"}
    if cleaned.lower() in ignored_words:
        return None
        
    # Verify the token starts with a valid digit or OCR-corrupted number substitute
    # to avoid false matching words like "total" (misread as "to1al" matching "o1")
    first_char = cleaned[0]
    if not re.match(r'[0-9.Ili|oO<>]', first_char):
        return None

    # A valid numeric string should match digits/decimals potentially corrupted by OCR (i.e. letters like o, O, l, I, |)
    # We require that it contains at least one actual digit to prevent pure text match
    match = re.search(r'[0-9Ili|oO]+(?:\.[0-9Ili|oO]+)?', cleaned)
    if match:
        candidate = match.group(0)
        if any(c.isdigit() for c in candidate):
            return candidate
            
    return None

def extract_biomarkers(text_lines: List[str]) -> Dict[str, Any]:
    """
    Extract CBC biomarkers from a list of layout-restructured lines.
    Prioritizes extracting the FIRST valid numeric value immediately following the biomarker name.
    """
    results = {
        "hemoglobin": None,
        "rbc": None,
        "wbc": None,
        "platelets": None
    }
    
    # Pre-process lines
    cleaned_lines = [clean_line(line) for line in text_lines if line.strip()]
    merged_lines = merge_broken_lines(cleaned_lines)
    
    logger.info(f"Processing {len(merged_lines)} lines for extraction")
    
    for biomarker, aliases in BIOMARKER_ALIASES.items():
        # Sort aliases by length descending to match longest alias first
        sorted_aliases = sorted(aliases, key=len, reverse=True)
        found_val = None
        
        for line in merged_lines:
            # Prior to checking matching, let's remove any obvious reference ranges and limit parameters
            # to make sure we don't accidentally capture them if the result column was misread.
            # 1. Remove range values like: 13.0 - 17.0, 4000-11000, 4.5-5.5
            range_pattern = r'\b[0-9Ili|oO]+(?:\.[0-9Ili|oO]+)?\s*(?:-|to|~)\s*[0-9Ili|oO]+(?:\.[0-9Ili|oO]+)?\b'
            line_cleaned = re.sub(range_pattern, '', line)
            
            # 2. Remove comparison limits like: < 11000, <= 4000, > 150000, etc.
            limit_pattern = r'[<>]=?\s*[0-9Ili|oO]+(?:\.[0-9Ili|oO]+)?\b'
            line_cleaned = re.sub(limit_pattern, '', line_cleaned)

            # Search for alias matching on the cleaned line
            matched_alias = None
            alias_match_obj = None
            
            for alias in sorted_aliases:
                # Use word boundary to avoid partial matches
                pattern = r'\b' + re.escape(alias) + r'\b'
                match = re.search(pattern, line_cleaned)
                if match:
                    matched_alias = alias
                    alias_match_obj = match
                    break
            
            if matched_alias and alias_match_obj:
                # Splice text after the matched alias
                after_text = line_cleaned[alias_match_obj.end():].strip()
                tokens = after_text.split()
                
                # Check tokens sequentially
                token_val = None
                for token in tokens:
                    parsed_candidate = try_parse_numeric(token)
                    if parsed_candidate:
                        cleaned_num = clean_numeric_string(parsed_candidate)
                        if cleaned_num:
                            token_val = cleaned_num
                            break
                            
                if token_val:
                    found_val = token_val
                    logger.info(f"Extracted {biomarker} value '{found_val}' from line: '{line}' via token parser")
                    break
                    
                # 2. Fallback: Check if number is directly merged with the alias (e.g. "hb12.5" or "hb:12.5")
                # We use a pattern matching the alias directly, followed by optional punctuation/spaces and the number
                fallback_pattern = re.escape(matched_alias) + r'\s*[:\-=]?\s*([0-9Ili|oO]+(?:\.[0-9Ili|oO]+)?)'
                fallback_match = re.search(fallback_pattern, line_cleaned)
                if fallback_match:
                    fallback_val = clean_numeric_string(fallback_match.group(1))
                    if fallback_val:
                        found_val = fallback_val
                        logger.info(f"Extracted {biomarker} value '{found_val}' from line: '{line}' via merged fallback")
                        break
        
        # 3. Post-process and cast value to expected types
        if found_val is not None:
            try:
                if biomarker in ["wbc", "platelets"]:
                    # cast float first in case it has decimal (e.g., 9000.0)
                    results[biomarker] = int(float(found_val))
                else:
                    results[biomarker] = float(found_val)
            except Exception as e:
                logger.error(f"Error casting {biomarker} value '{found_val}': {e}")
                results[biomarker] = None
                
    return results
