import sys
import os
import glob
from typing import Dict, List
from pathlib import Path
import pdfplumber

from quest.core.datapack.doc import TextDoc, ZenDBDoc

# SHT
def util_load_zendb_docs(paths, debug_flag=False, topK = 1, start_doc_id = 1) -> List[ZenDBDoc]:
    """vision"""
    print("\n parse...")
    docs = []
    for i, path in enumerate(paths):
        if debug_flag and i >= topK: 
            print(f"Debug : only for {topK} documents")
            break
        ext = Path(path).suffix.lower()
        # print(f"  process {i+1}/{len(paths)}: {Path(path).name}")
        print(f" filename: {Path(path).name}, doc_id: {i+start_doc_id}: ") # (name, doc_id)

        try:
            if ext == ".pdf":
                with pdfplumber.open(path) as pdf:
                    text = ""
                    all_words = []
                    for page_num, page in enumerate(pdf.pages):
                        words = page.extract_words(extra_attrs=["fontname", "size", "x0", "y0"])
                        all_words.extend(words)
                        page_text = page.extract_text()
                        if page_text:
                            text += page_text + "\n"
                doc = ZenDBDoc(attr_dict = {
                    "id": f"doc_{i+start_doc_id}",
                    "doc_id": i+start_doc_id,
                    "path": path,
                    "name": Path(path).name,  # .stem
                    "text": text,
                    "words": all_words,
                    "num_pages": len(pdf.pages)
                }) 
            elif ext == ".txt" or ext == ".md":
                with open(path, "r", encoding="utf-8") as f:
                    text = f.read()
                doc = ZenDBDoc(attr_dict = {
                    "id": f"doc_{i}",
                    "doc_id": i+start_doc_id,
                    "path": path,
                    "name": Path(path).name,
                    "text": text,
                    "words": [],  # txt
                    "num_pages": 1
                }) 
            else:
                print(f"not support document type: {ext}")
                continue
            docs.append(doc)
            print(f"    extract {len(doc['words'])} words，{len(doc['text'])} characters")
        except Exception as e:
            print(f"    failed: {e}")
            continue
    return docs

def load_ZenDBDoc_from_directory(docs_dir: str, table_name: str, start_doc_id = 1, debug_flag = False) -> List[ZenDBDoc]:
    docs = []
    print(f"Loading documents from {docs_dir}...")
    # doc_id = start_doc_id
    
    # get all .txt and .pdf, merge
    txt_files = glob.glob(os.path.join(docs_dir, "*.txt"))
    pdf_files = glob.glob(os.path.join(docs_dir, "*.md")) 

    txt_files.extend(pdf_files)
    txt_files.sort()  # make sure the order
    
    print(f"Loading documents from {docs_dir}...")
    print(f"Found {len(txt_files)} text files") 
    if debug_flag:
        print(f"debug for one doc")
        txt_files = txt_files[0:1]
    docs = util_load_zendb_docs(txt_files, debug_flag=debug_flag, topK=5, start_doc_id = start_doc_id) 
    next_doc_id = start_doc_id + len(docs) 
    
    return docs, next_doc_id




def load_TextDocs_from_directory(docs_dir: str, table_name: str, start_doc_id = 1, debug_flag = False) -> List[TextDoc]:
    """
    load docs
    
    Args:
        docs_dir: 
        table_name: to get doc_id
        
    Returns:
        TextDoc List
    """
    docs = []
    doc_id = start_doc_id
    
    # get all .txt
    txt_files = glob.glob(os.path.join(docs_dir, "*.txt"))
    txt_files.sort()  # order
    
    print(f"Loading documents from {docs_dir}...")
    print(f"Found {len(txt_files)} text files")
    
    for file_path in txt_files:
        if debug_flag and len(docs) >= 2:  # debug
            print(f"debug for 2 docs")
            break
        try:
            # read
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # get filename
            file_name = os.path.basename(file_path)
            
            # carete TextDoc, metadata includes file_name
            doc = TextDoc(
                content=content,
                doc_id=doc_id,
                metadata={"file_name": file_name, "table": table_name}
            )
            
            docs.append(doc)
            doc_id += 1
            
            print(f"  Loaded: {file_name} (doc_id: {doc.doc_id})")
            
        except Exception as e:
            print(f"Error loading file {file_path}: {e}")
            continue
    
    print(f"Successfully loaded {len(docs)} documents from {table_name}")
    next_doc_id = doc_id
    return docs, next_doc_id
