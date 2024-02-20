import os
from PyPDF2 import PdfReader

def split_pdf_into_chunks(pdf_path, chunk_size=1000):
    # Open the PDF file
    with open(pdf_path, 'rb') as file:
        reader = PdfReader(file)
        num_pages = len(reader.pages) 

        # Loop through each page and extract text
        for page_num in range(num_pages):
            page = reader.pages[page_num]
            text = page.extract_text()

            # Split text into chunks
            chunks = [text[i:i+chunk_size] for i in range(0, len(text), chunk_size)]

            # Write chunks to separate files
            for i, chunk in enumerate(chunks):
                chunk_filename = f"{os.path.splitext(pdf_path)[0]}_page{page_num+1}_chunk{i+1}.txt"
                with open(chunk_filename, 'w', encoding='utf-8') as chunk_file:
                    chunk_file.write(chunk)

if __name__ == "__main__":
    # Specify the folder containing PDF files
    pdf_folder = '/home/enterpi/Desktop/PdfReader/AA Articles'

    # Iterate through each PDF file in the folder
    for filename in os.listdir(pdf_folder):
        if filename.endswith('.pdf'):
            pdf_path = os.path.join(pdf_folder, filename)
            split_pdf_into_chunks(pdf_path)





