import re
import argparse
import random
import os  # Added for file operations
import json  # Added for JSON operations
from bs4 import BeautifulSoup
from collections import Counter, defaultdict

# NEW ADDITION: Comprehensive synonym dictionary for finance dataset attributes
# This was not present in the old version
ATTRIBUTE_SYNONYMS = {
   # Finance dataset attributes
    "company_name": [
        "company", "corporation", "limited", "ltd", "pty", "inc", "incorporated", "technologies",
        "group", "holdings", "enterprise", "business", "firm", "entity", "organization",
        r"\b([A-Z][a-z]+ Ltd)\b", r"\b([A-Z][a-z]+ Limited)\b", r"\b([A-Z][a-z]+ Pty Ltd)\b",
        r"\b([A-Z][a-z]+ Corporation)\b", r"\b([A-Z][a-z]+ Technologies)\b", r"\b([A-Z][a-z]+ Group)\b"
    ],
    "registered_office": [
        "registered office", "head office", "principal office", "main office", "corporate office",
        "address", "location", "headquarters", "domicile", "registered address",
        r"\b(\d+ [A-Z][a-z]+ Street)\b", r"\b([A-Z][a-z]+, [A-Z][A-Z] \d+)\b",
        r"\b(Suite \d+)\b", r"\b(Level \d+)\b", r"\b(PO Box \d+)\b"
    ],
    "exchange_code": [
        "exchange", "stock exchange", "listed", "asx", "nasdaq", "nyse", "lse", "trading",
        "ticker", "symbol", "code", "market", "securities exchange",
        r"\b(ASX)\b", r"\b(NYSE)\b", r"\b(NASDAQ)\b", r"\b(LSE)\b", r"\b([A-Z]{3,4})\b"
    ],
    "principal_activities": [
        "principal activities", "main activities", "business activities", "operations",
        "industry", "sector", "business", "activities", "services", "products",
        "mining", "finance", "healthcare", "manufacturing", "technology", "retail",
        "energy", "utilities", "real estate", "transportation", "agriculture",
        "telecommunications", "media", "other"
    ],
    "board_members": [
        "board", "directors", "board of directors", "board members", "director",
        "chairman", "chairperson", "non-executive", "executive", "independent",
        r"\b(Mr\. [A-Z][a-z]+ [A-Z][a-z]+)\b", r"\b(Ms\. [A-Z][a-z]+ [A-Z][a-z]+)\b",
        r"\b(Chairman)\b", r"\b(Director)\b", r"\b(Board Member)\b"
    ],
    "executive_profiles": [
        "executive", "management", "executive team", "senior management", "ceo", "cfo",
        "coo", "managing director", "chief", "officer", "manager", "executive management",
        r"\b(CEO)\b", r"\b(CFO)\b", r"\b(COO)\b", r"\b(Chief Executive Officer)\b",
        r"\b(Managing Director)\b", r"\b(Executive)\b"
    ],
    "revenue": [
        "revenue", "income", "sales", "turnover", "gross income", "operating revenue",
        "total revenue", "net sales", "receipts", "earnings",
        r"\b(\$[\d,]+)\b", r"\b(\d+,\d+)\b", r"\b(million)\b", r"\b(thousand)\b"
    ],
    "net_profit_or_loss": [
        "net profit", "net loss", "profit", "loss", "net income", "earnings",
        "profit after tax", "net result", "bottom line", "net earnings",
        r"\b(\$[\d,]+)\b", r"\b(\(\$[\d,]+\))\b", r"\b(loss)\b", r"\b(profit)\b"
    ],
    "total_debt": [
        "total debt", "debt", "liabilities", "borrowings", "loans", "total liabilities",
        "financial debt", "interest bearing debt", "bank loans", "credit facilities",
        r"\b(\$[\d,]+)\b", r"\b(debt)\b", r"\b(loan)\b", r"\b(borrowing)\b"
    ],
    "total_assets": [
        "total assets", "assets", "total resources", "balance sheet total",
        "current assets", "non-current assets", "fixed assets", "investments",
        r"\b(\$[\d,]+)\b", r"\b(assets)\b", r"\b(property)\b", r"\b(equipment)\b"
    ],
    "cash_reserves": [
        "cash", "cash equivalents", "cash reserves", "liquid assets", "bank balances",
        "short-term deposits", "money market", "treasury", "cash on hand",
        r"\b(\$[\d,]+)\b", r"\b(cash)\b", r"\b(deposits)\b", r"\b(liquid)\b"
    ],
    "net_assets": [
        "net assets", "equity", "shareholders equity", "net worth", "total equity",
        "stockholders equity", "owners equity", "capital", "retained earnings",
        r"\b(\$[\d,]+)\b", r"\b(equity)\b", r"\b(net worth)\b", r"\b(capital)\b"
    ],
    "earnings_per_share": [
        "earnings per share", "eps", "basic eps", "diluted eps", "per share earnings",
        "share earnings", "earnings ratio", "profit per share",
        r"\b(\$?[\d.]+)\b", r"\b(cents)\b", r"\b(per share)\b", r"\b(EPS)\b"
    ],
    "dividend_per_share": [
        "dividend", "dividend per share", "dividends paid", "dividend declared",
        "dividend payment", "interim dividend", "final dividend", "special dividend",
        r"\b(\$?[\d.]+)\b", r"\b(cents)\b", r"\b(dividend)\b", r"\b(per share)\b"
    ],
    "largest_shareholder": [
        "largest shareholder", "major shareholder", "principal shareholder", "controlling shareholder",
        "substantial shareholder", "significant shareholder", "top shareholder", "main investor",
        r"\b([A-Z][a-z]+ [A-Z][a-z]+)\b", r"\b([A-Z][a-z]+ Group)\b", r"\b([A-Z][a-z]+ Holdings)\b"
    ],
    "the_highest_ownership_stake": [
        "ownership", "shareholding", "stake", "percentage", "percent", "holding",
        "interest", "share", "equity interest", "voting rights",
        r"\b(\d+\.?\d*%)\b", r"\b(\d+\.?\d* percent)\b", r"\b(\d+\.?\d* per cent)\b"
    ],
    "major_equity_changes": [
        "equity changes", "capital changes", "share issues", "share buyback",
        "rights issue", "capital raising", "merger", "acquisition", "restructuring",
        "yes", "no", "major changes", "significant changes"
    ],
    "major_events": [
        "major events", "significant events", "corporate events", "material events",
        "m&a", "merger", "acquisition", "litigation", "lawsuit", "major contract",
        "leadership change", "restructuring", "delisting", "other", "material matters"
    ],
    "bussiness_sales": [
        "business sales", "segment revenue", "division sales", "operating segments",
        "business lines", "revenue breakdown", "sales by segment", "operating revenue",
        r"\b(\$[\d,]+)\b", r"\b(segment)\b", r"\b(division)\b", r"\b(business unit)\b"
    ],
    "bussiness_profit": [
        "business profit", "segment profit", "division profit", "operating profit",
        "segment earnings", "profit by segment", "divisional profit", "business unit profit",
        r"\b(\$[\d,]+)\b", r"\b(segment)\b", r"\b(profit)\b", r"\b(earnings)\b"
    ],
    "bussiness_cost": [
        "business cost", "segment cost", "operating costs", "cost of sales",
        "divisional costs", "business expenses", "segment expenses", "operating expenses",
        r"\b(\$[\d,]+)\b", r"\b(cost)\b", r"\b(expense)\b", r"\b(expenditure)\b"
    ],
    "business_segments_num": [
        "business segments", "operating segments", "reportable segments", "divisions",
        "business units", "segment reporting", "number of segments",
        r"\b(\d+)\b", r"\b(segment)\b", r"\b(division)\b", r"\b(business unit)\b"
    ],
    "business_risks": [
        "business risks", "risk factors", "principal risks", "key risks", "material risks",
        "market risk", "credit risk", "operational risk", "legal risk", "compliance risk",
        "environmental risk", "strategic risk", "other", "risk management"
    ],
    "remuneration_policy": [
        "remuneration", "compensation", "executive pay", "director fees", "salary",
        "fixed", "performance-based", "stock option", "equity", "mixed", "incentive",
        "bonus", "not disclosed", "remuneration policy", "pay structure"
    ],
    "auditor": [
        "auditor", "audit firm", "external auditor", "independent auditor", "accountant",
        "audit", "pwc", "deloitte", "kpmg", "ey", "ernst & young", "grant thornton",
        r"\b([A-Z][a-z]+ [A-Z][a-z]+ Audit)\b", r"\b([A-Z][a-z]+ Thornton)\b"
    ]
}




def set_profiler_args(profiler_args):

    parser = argparse.ArgumentParser(
        "LLM profiler.",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--chunk_size",
        type=int,
        default=5000
    )

    parser.add_argument(
        "--train_size",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--eval_size",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--max_chunks_per_file",
        type=int,
        default=-1,
    )

    parser.add_argument(
        "--num_top_k_scripts",
        type=int,
        default=1,
        help="of all the scripts we generate for the metadata fields, number to retain after scoring their qualities",
    )

    parser.add_argument(
        "--extraction_fraction_thresh",
        type=int,
        default=0.9,
        help="for abstensions approach",
    )

    parser.add_argument(
        "--remove_tables",
        type=bool,
        default=False,
        help="Remove tables from the html files?",
    )

    parser.add_argument(
        "--body_only",
        type=bool,
        default=False,
        help="Only use HTML body",
    )

    parser.add_argument(
        "--max_metadata_fields",
        type=int,
        default=15,
    )

    parser.add_argument(
        "--use_dynamic_backoff",
        type=bool,
        default=True,
        help="Whether to do the function generation workflow or directly extract from chunks",
    )

    parser.add_argument(
        "--use_qa_model",
        type=bool,
        default=False,
        help="Whether to apply the span-extractor QA model.",
    )

    parser.add_argument(
        "--overwrite_cache", 
        type=int, 
        default=0,
        help="overwrite the manifest cache"
    )

    # models to use in the pipeline
    parser.add_argument(
        "--MODELS", 
        type=list, 
        help="models to use in the pipeline"
    )

    parser.add_argument(
        "--KEYS", 
        type=list, 
        help="keys for openai models"
    )

    parser.add_argument(
        "--GOLDKEY", 
        type=str, 
        help="models to use in the pipeline"
    )

    parser.add_argument(
        "--MODEL2URL",
        type=dict,
        default={"llama-3-1-70B": "https://aihubmix.com/v1"},
        help="models to use in the pipeline"
    )

    parser.add_argument(
        "--swde_plus",
        type=bool,
        default=False,
        help="Whether to use the extended SWDE dataset to measure OpenIE performance",
    )

    parser.add_argument(
        "--schema_id_sizes",
        type=int,
        default=0,
        help="Number of documents to use for schema identification stage, if it differs from extraction",
    )

    parser.add_argument(
        "--slice_results",
        type=bool,
        default=False,
        help="Whether to measure the results by attribute-slice",
    )

    parser.add_argument(
        "--fn_generation_prompt_num",
        type=int,
        default=-1,
        help="For ablations on function generation with diversity, control which prompt we use. Default is all.",
    )

    parser.add_argument(
        "--upper_bound_fns",
        type=bool,
        default=False,
        help="For ablations that select functions using ground truth instead of the FM.",
    )

    parser.add_argument(
        "--combiner_mode",
        type=str,
        default='mv',
        help="For ablations that select functions using ground truth instead of the FM.",
    )

    parser.add_argument(
        "--use_alg_filtering",
        type=str,
        default=True,
        help="Whether to filter functions based on quality.",
    )

    parser.add_argument(
        "--use_abstension",
        type=str,
        default=True,
        help="Whether to use the abstensions approach.",
    )

    args = parser.parse_args(args=[])
    for arg, val in profiler_args.items():
        setattr(args, arg, val)
    return args


#################### GET SOME SAMPLE FILES TO SEED THE METADATA SEARCH #########################

def sample_scripts(files, train_size=5):
    """
    Select files based on table.json entries.
    For legal case dataset, select the first train_size files that have complete data in table.json.
    """
    # OLD VERSION:
    # # "Train" split
    # random.seed(0)
    # if train_size <= len(files):
    #     sample_files = random.sample(files, train_size)
    # else:
    #     sample_files = files
    # sample_contents = []
    # for sample_file in sample_files:
    #     with open(sample_file, 'r',errors='ignore') as f:
    #         sample_contents.append(f.read())
    # return sample_files
    
    import json
    import os
    
    # Try to find table.json file in multiple possible locations
    table_json_path = None
    
    # Strategy 1: Check if we have files and try to derive the base path
    if files:
        sample_file = files[0]
        # From: /home/guyang/evaporate/data/evaporate/data/lcr/data/evaporate/swde/movie/legal_case/xxx.txt
        # To:   /home/guyang/evaporate/data/evaporate/data/lcr/table.json
        
        # Try going up to find the lcr directory
        current_dir = os.path.dirname(sample_file)
        while current_dir and current_dir != '/':
            if 'lcr' in current_dir:
                # Found lcr directory, check for table.json here
                potential_table_path = os.path.join(current_dir, 'table.json')
                if os.path.exists(potential_table_path):
                    table_json_path = potential_table_path
                    break
                # Also check parent directories of lcr
                parent_lcr = os.path.dirname(current_dir)
                if 'lcr' in os.path.basename(parent_lcr):
                    potential_table_path = os.path.join(parent_lcr, 'table.json')
                    if os.path.exists(potential_table_path):
                        table_json_path = potential_table_path
                        break
            current_dir = os.path.dirname(current_dir)
    
    # Strategy 2: Try common locations
    if not table_json_path:
        common_paths = [
            "/home/guyang/evaporate/data/evaporate/data/finance/table.json",
            "data/evaporate/data/lcr/table.json",
            "table.json"
        ]
        for path in common_paths:
            if os.path.exists(path):
                table_json_path = path
                break
    
    if not table_json_path:
        # Fallback to original logic if table.json not found
        print("table.json not found, using original file selection logic")
        if train_size <= len(files):
            def extract_number(filepath):
                import re
                match = re.search(r'/(\d+)\.txt$', filepath)
                return int(match.group(1)) if match else float('inf')
            
            sorted_files = sorted(files, key=extract_number)
            sample_files = sorted_files[:train_size]
        else:
            sample_files = files
    else:
        # Load table.json and select files with complete data
        try:
            with open(table_json_path, 'r') as f:
                table_data = json.load(f)
            
            print(f"Found table.json at: {table_json_path}")
            print(f"Table.json contains {len(table_data)} entries")
            print("Selecting first 10 entries from table.json in numeric order")
            
            # 按数字顺序排序 table.json 的键
            def extract_number_from_path(file_path):
                import re
                match = re.search(r'/(\d+)\.txt$', file_path)
                return int(match.group(1)) if match else float('inf')
            
            # 按文件名中的数字排序
            sorted_paths = sorted(table_data.keys(), key=extract_number_from_path)
            
            # 选择前 train_size 个
            selected_file_paths = sorted_paths[:train_size]
            
            print(f"DEBUG: Selected table.json keys: {[os.path.basename(p) for p in selected_file_paths]}")
            
            # 修复文件匹配逻辑 - 使用精确匹配而不是 endswith
            sample_files = []
            for selected_path in selected_file_paths:
                file_name = os.path.basename(selected_path)  # 获取文件名：1.txt
                
                # 找到匹配的文件 - 使用精确的文件名匹配
                matched = False
                for file in files:
                    if os.path.basename(file) == file_name:  # 精确匹配文件名
                        sample_files.append(file)
                        matched = True
                        break
                
                if not matched:
                    print(f"Warning: Could not find file for {file_name}")
            
            print(f"Selected files from table.json: {[os.path.basename(f) for f in sample_files]}")
            
        except Exception as e:
            print(f"Error loading table.json: {e}, using original logic")
            # Fallback to original logic
            if train_size <= len(files):
                def extract_number(filepath):
                    import re
                    match = re.search(r'/(\d+)\.txt$', filepath)
                    return int(match.group(1)) if match else float('inf')
                
                sorted_files = sorted(files, key=extract_number)
                sample_files = sorted_files[:train_size]
            else:
                sample_files = files
    
    sample_contents = []
    for sample_file in sample_files:
        try:
            with open(sample_file, 'r') as f:
                sample_contents.append(f.read())
        except Exception as e:
            print(f"Error reading file {sample_file}: {e}")
    
    print(f"Selected training files: {[f.split('/')[-1] for f in sample_files[:10]]}")
    return sample_files
 

#################### BOILERPLATE CHUNKING CODE, CRITICAL FOR LONG SEUQENCES ####################
def chunk_file(
    parser, file, chunk_size=5000, mode="train", remove_tables=False, body_only=False
):
    content =  get_file_contents(file)
    if "html" in parser:
        content, chunks = get_html_parse(
            content, 
            chunk_size=chunk_size, 
            mode=mode, 
            remove_tables=remove_tables,
            body_only=body_only
        )
    else:
        content, chunks = get_txt_parse(content, chunk_size=chunk_size, mode=mode)
    return content, chunks


# HTML --> CHUNKS
def clean_html(content):
    for tag in ['script', 'style', 'svg']:
        content = content.split("\n")
        clean_content = []
        in_script = 0
        for c in content:
            if c.strip().strip("\t").startswith(f"<{tag}"):
                in_script = 1
            endstr = "</" + tag # + ">"
            if endstr in c or "/>" in c:
                in_script = 0
            if not in_script:
                clean_content.append(c)
        content = "\n".join(clean_content)
    return content


def get_flattened_items(content, chunk_size=500):
    flattened_divs = str(content).split("\n")
    flattened_divs = [ch for ch in flattened_divs if ch.strip() and ch.strip("\n").strip()]

    clean_flattened_divs = []
    for div in flattened_divs:
        if len(str(div)) > chunk_size:
            sub_divs = div.split("><")
            if len(sub_divs) == 1:
                clean_flattened_divs.append(div)
            else:
                clean_flattened_divs.append(sub_divs[0] + ">")
                for sd in sub_divs[1:-1]:
                    clean_flattened_divs.append("<" + sd + ">")
                clean_flattened_divs.append("<" + sub_divs[-1])
        else:
            clean_flattened_divs.append(div)
    return clean_flattened_divs
 

def get_html_parse(content, chunk_size=5000, mode="train", remove_tables=False, body_only=False):
    if remove_tables:
        soup = BeautifulSoup(content)
        tables = soup.find_all("table")
        for table in tables:
            if "infobox" not in str(table):
                content = str(soup)
                content = content.replace(str(table), "")
                soup = BeautifulSoup(content)

    if body_only:
        soup = BeautifulSoup(content)
        content = str(soup.find("body"))
        soup = BeautifulSoup(content)

    else:
        content = clean_html(content)
        clean_flattened_divs = []
        flattened_divs = get_flattened_items(content, chunk_size=chunk_size)
        for i, div in enumerate(flattened_divs):
            new_div = re.sub(r'style="[^"]*"', '', str(div))
            new_div = re.sub(r'<style>.*?</style>', '', str(new_div))
            new_div = re.sub(r'<style.*?/style>', '', str(new_div))
            new_div = re.sub(r'<meta.*?/>', '', str(new_div))
            new_div = "\n".join([l for l in new_div.split("\n") if l.strip() and l.strip("\n").strip()])
            # new_div = BeautifulSoup(new_div) #.fsind_all("div")[0]
            if new_div:
                clean_flattened_divs.append(new_div)

        if mode == "eval":
            return []

    grouped_divs = []
    current_div = []
    current_length = 0
    max_length = chunk_size
    use_raw_text = False  # Added to fix undefined variable
    join_str = " " if use_raw_text else "\n"
    for div in clean_flattened_divs:
        str_div = str(div)
        len_div = len(str_div)
        if (current_length + len_div > max_length):
            grouped_divs.append(join_str.join(current_div))
            current_div = []
            current_length = 0
        elif not current_div and (current_length + len_div > max_length):
            grouped_divs.append(str_div)
            continue
        current_div.append(str_div)
        current_length += len_div

    return content, grouped_divs


# GENERIC TXT --> CHUNKS
def get_txt_parse(content, chunk_size=5000, mode="train"):
    # convert to chunks
    if mode == "train":
        chunks = content.split("\n")
        clean_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                sub_chunks = chunk.split(". ")
                clean_chunks.extend(sub_chunks)
            else:
                clean_chunks.append(chunk)

        chunks = clean_chunks.copy()
        clean_chunks = []
        for chunk in chunks:
            if len(chunk) > chunk_size:
                sub_chunks = chunk.split(", ")
                clean_chunks.extend(sub_chunks)
            else:
                clean_chunks.append(chunk)

        final_chunks = []
        cur_chunk = []
        cur_chunk_size = 0
        for chunk in clean_chunks:
            if cur_chunk_size + len(chunk) > chunk_size:
                final_chunks.append("\n".join(cur_chunk))
                cur_chunk = []
                cur_chunk_size = 0
            cur_chunk.append(chunk)
            cur_chunk_size += len(chunk)
        if cur_chunk:
            final_chunks.append("\n".join(cur_chunk))
    else:
        final_chunks = []
    return content, final_chunks


def get_file_contents(file):
    text = ''
    if file.endswith(".swp"):
        return text
    try:
        with open(file) as f:
            text = f.read()
    except:
        with open(file, "rb") as f:
            text = f.read().decode("utf-8", "ignore")
    return text


def clean_metadata(field):
    return field.replace("\t", " ").replace("\n", " ").strip().lower()


def match_with_synonyms(attribute, chunk):
    # NEW FUNCTION ADDED: Enhanced attribute matching with synonyms from ATTRIBUTE_SYNONYMS dictionary
    # 取同义词列表，若无则只用原属性名
    synonyms = ATTRIBUTE_SYNONYMS.get(attribute.lower(), [attribute])
    for syn in synonyms:
        if syn.startswith(r"\b"):  # 正则表达式
            if re.search(syn, chunk):
                return True
        else:
            if syn.lower() in chunk.lower():
                return True
    return False

def filter_file2chunks(file2chunks, sample_files, attribute):
    def get_attribute_parts(attribute):
        for char in ["/", "-", "(", ")", "[", "]", "{", "}", ":"]:
            attribute = attribute.replace(char, " ")
        attribute_parts = attribute.lower().split()
        return attribute_parts

    # filter chunks with simple keyword search
    attribute_chunks = defaultdict(list)
    starting_num_chunks = 0
    ending_num_chunks = 0
    ending_in_sample_chunks = 0
    starting_in_sample_chunks = 0
    for file, chunks in file2chunks.items():
        starting_num_chunks += len(chunks)
        if file in sample_files:
            starting_in_sample_chunks += len(chunks)
        cleaned_chunks = []
        
        # 在filter_file2chunks里替换原有查找
        for chunk in chunks:
            if match_with_synonyms(attribute, chunk):
                cleaned_chunks.append(chunk)
        # OLD VERSION:
        # for chunk in chunks:
        #     if attribute.lower() in chunk.lower():
        #         cleaned_chunks.append(chunk)
        if len(cleaned_chunks) == 0:
            for chunk in chunks:
                if attribute.lower().replace(" ", "") in chunk.lower().replace(" ", ""):
                    cleaned_chunks.append(chunk)
        if len(cleaned_chunks) == 0:
            chunk2num_word_match = Counter()
            for chunk_num, chunk in enumerate(chunks):
                attribute_parts = get_attribute_parts(attribute.lower())
                for wd in attribute_parts:
                    if wd.lower() in chunk.lower():
                        chunk2num_word_match[chunk_num] += 1
            # sort chunks by number of words that match
            sorted_chunks = sorted(chunk2num_word_match.items(), key=lambda x: x[1], reverse=True)
            if len(sorted_chunks) > 0:
                cleaned_chunks.append(chunks[sorted_chunks[0][0]])
            if len(sorted_chunks) > 1:
                cleaned_chunks.append(chunks[sorted_chunks[1][0]])
        ending_num_chunks += len(cleaned_chunks)
        num_chunks = len(cleaned_chunks)
        num_chunks = min(num_chunks, 2)

        cleaned_chunks = cleaned_chunks[:num_chunks]
        attribute_chunks[file] = cleaned_chunks
        if file in sample_files:
            ending_in_sample_chunks += len(attribute_chunks[file])
    file2chunks = attribute_chunks
    if ending_num_chunks == 0 or ending_in_sample_chunks == 0:
        print(f"Removing because no chunks for attribute {attribute} in any file")
        return None
    print(f"For attribute {attribute}\n-- Starting with {starting_num_chunks} chunks\n-- Ending with {ending_num_chunks} chunks")
    print(f"-- {starting_in_sample_chunks} starting chunks in sample files\n-- {ending_in_sample_chunks} chunks in sample files")

    return file2chunks


def clean_function_predictions(extraction, attribute=None):
        if extraction is None:
            return ''
        if type(extraction) == list:
            if extraction and type(extraction[0]) == list:
                full_answer = []
                for answer in extraction:
                        if type(answer) == list:
                            dedup_list = []
                            for a in answer:
                                if a not in dedup_list:
                                    dedup_list.append(a)
                            answer = dedup_list
                            answer = [str(a).strip().strip("\n") for a in answer]
                            full_answer.append(", ".join(answer))
                        else:
                            full_answer.append(answer.strip().strip("\n"))
                full_answer = [a.strip() for a in full_answer]
                extraction = ", ".join(full_answer)
            elif extraction and len(extraction) == 1 and extraction[0] is None:
                extraction = ''
            else:
                dedup_list = []
                for a in extraction:
                    if a not in dedup_list:
                        dedup_list.append(a)
                extraction = dedup_list
                extraction = [(str(e)).strip().strip("\n") for e in extraction]
                extraction = ", ".join(extraction)
        if type(extraction) == "str" and extraction.lower() == "none":
            extraction = ""
        extraction = extraction.strip().replace("  ", " ")
        if extraction.lower().startswith(attribute.lower()):
            idx = extraction.lower().find(attribute.lower())
            extraction = extraction[idx+len(attribute):].strip()
        for char in [':', ","]:
            extraction = extraction.strip(char).strip()
        extraction  = extraction.replace(",", ", ").replace("  ", " ")
        return extraction


def check_vs_train_extractions(train_extractions, final_extractions, gold_key, attribute = None):
        clean_final_extractions = {}

        gold_values = train_extractions[gold_key]
        modes = []
        start_toks = []
        end_toks = []
        for file, gold in gold_values.items():
            if type(gold) == dict:
                gold = gold[attribute]
            if type(gold) == list:
                if gold and type(gold[0]) == list:
                    gold = [g[0] for g in gold]
                    gold = ", ".join(gold)
                else:
                    gold = ", ".join(gold)
            gold = gold.lower()
            pred = final_extractions[file].lower()
            if not pred or not gold:
                continue
            if ("<" in pred and "<" not in gold) or (">" in pred and ">" not in gold):
                check_pred =  BeautifulSoup(pred).text
                if check_pred in gold or gold in check_pred:
                    modes.append("soup")
            elif gold in pred and len(pred) > len(gold):
                modes.append("longer")
                idx = pred.index(gold)
                if idx > 0:
                    start_toks.append(pred[:idx-1])
                end_idx = idx + len(gold)
                if end_idx < len(pred):
                    end_toks.append(pred[end_idx:])

        def long_substr(data):
            substr = ''
            if len(data) > 1 and len(data[0]) > 0:
                for i in range(len(data[0])):
                    for j in range(len(data[0])-i+1):
                        if j > len(substr) and is_substr(data[0][i:i+j], data):
                            substr = data[0][i:i+j]
            return substr

        def is_substr(find, data):
            if len(data) < 1 and len(find) < 1:
                return False
            for i in range(len(data)):
                if find not in data[i]:
                    return False
            return True
        
        longest_end_tok = long_substr(end_toks)
        longest_start_tok = long_substr(start_toks)
        if len(set(modes)) == 1:
            num_golds = len(gold_values)
            for file, extraction in final_extractions.items():
                if "longer" in modes:
                    # gold longer than pred
                    if len(end_toks) == num_golds and longest_end_tok in extraction and extraction.count(longest_end_tok) == 1:
                        idx = extraction.index(longest_end_tok)
                        extraction = extraction[:idx] 
                    if len(start_toks) == num_golds and longest_start_tok in extraction and extraction.count(longest_start_tok) == 1:
                        idx = extraction.index(longest_start_tok)
                        extraction = extraction[idx:] 
                elif "soup" in modes:
                    extraction = BeautifulSoup(extraction).text
                clean_final_extractions[file] = extraction
        else:
            clean_final_extractions = final_extractions
        return clean_final_extractions
