#!/usr/bin/env python3
"""
Create a better chunking strategy for test documents
"""

def chunk_text_by_sections(text: str, max_chunk_size: int = 300) -> list:
    """
    Chunk text by sections/paragraphs, ensuring each semantic unit stays together
    """
    chunks = []
    
    # Split by double newlines first (paragraphs)
    paragraphs = text.split('\n\n')
    
    current_chunk = ""
    
    for paragraph in paragraphs:
        paragraph = paragraph.strip()
        if not paragraph:
            continue
            
        # If paragraph is too long, split by sentences
        if len(paragraph) > max_chunk_size:
            sentences = paragraph.split('. ')
            for sentence in sentences:
                if not sentence.strip():
                    continue
                sentence = sentence.strip() + '.'
                
                if len(current_chunk) + len(sentence) + 1 <= max_chunk_size:
                    if current_chunk:
                        current_chunk += " " + sentence
                    else:
                        current_chunk = sentence
                else:
                    if current_chunk:
                        chunks.append(current_chunk)
                    current_chunk = sentence
        else:
            # If adding this paragraph exceeds limit, save current chunk
            if current_chunk and len(current_chunk) + len(paragraph) + 1 > max_chunk_size:
                chunks.append(current_chunk)
                current_chunk = paragraph
            else:
                if current_chunk:
                    current_chunk += "\n\n" + paragraph
                else:
                    current_chunk = paragraph
    
    # Add any remaining content
    if current_chunk:
        chunks.append(current_chunk)
    
    return chunks

# Test with filing info
test_text = """Michigan Minor Guardianship - Filing Information Test Document

Filing Fees in Genesee County:
- The standard filing fee for a minor guardianship petition is $175.00
- If you cannot afford the filing fee, you may request a fee waiver using Form MC 20
- The court will review your financial situation to determine eligibility

Court Location:
Genesee County Probate Court
900 S. Saginaw Street
Flint, MI 48502

Hearing Schedule:
- Guardianship hearings in Genesee County are held on Thursdays
- Hearings typically start at 9:00 AM
- You must arrive at least 15 minutes early to check in

Required Forms:
- Form PC 651: Petition for Appointment of Guardian of a Minor
- Form PC 652: Notice of Hearing
- Additional forms may be required based on your specific situation

This is a test document for integration testing purposes."""

chunks = chunk_text_by_sections(test_text, max_chunk_size=300)
print(f"Created {len(chunks)} chunks:\n")
for i, chunk in enumerate(chunks):
    print(f"Chunk {i+1} ({len(chunk)} chars):")
    print(f"{chunk[:100]}..." if len(chunk) > 100 else chunk)
    print()