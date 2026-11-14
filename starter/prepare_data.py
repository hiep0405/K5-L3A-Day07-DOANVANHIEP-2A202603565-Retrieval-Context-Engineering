import chromadb
import json
import re
from pathlib import Path

# 1. Connect to ChromaDB and extract chunks from Strategy B
client = chromadb.PersistentClient(path='./DAY06_VIN_AI/K5-L3-Day06-2A202603565-DataIngestionEngineering/chroma_db')
col = client.get_collection('collection_strategy_b')
data = col.get()

chunks = []
chunks_by_page = {}
for cid, doc, meta in zip(data['ids'], data['documents'], data['metadatas']):
    page = int(meta.get('page', 1))
    c_obj = {
        'id': cid,
        'source_id': meta.get('document_id', 'bo-luat-lao-dong-2019'),
        'source': meta.get('source', 'bo_luat_lao_dong_2019_clean.pdf'),
        'page': page,
        'text': doc,
        'metadata': {
            'chapter': meta.get('chapter', ''),
            'strategy': meta.get('strategy', 'B')
        }
    }
    chunks.append(c_obj)
    chunks_by_page.setdefault(page, []).append(c_obj)

print(f"Extracted {len(chunks)} chunks.")

# 2. Load Day 6 questions
with open('./DAY06_VIN_AI/K5-L3-Day06-2A202603565-DataIngestionEngineering/benchmark/questions.json', encoding='utf-8') as f:
    d6_questions = json.load(f)

def find_verbatim_evidence(raw_quote, chunk_text):
    if raw_quote in chunk_text:
        return raw_quote
    words = raw_quote.strip().split()
    if not words:
        return None
    # Flexible whitespace
    pattern = r'\s+'.join(re.escape(w) for w in words)
    m = re.search(pattern, chunk_text)
    if m:
        return m.group(0)
    # Sliding window if needed
    for length in range(len(words), 4, -1):
        for start in range(len(words) - length + 1):
            sub_words = words[start:start + length]
            pattern = r'\s+'.join(re.escape(w) for w in sub_words)
            m = re.search(pattern, chunk_text)
            if m:
                return m.group(0)
    return None

new_questions = []

for q in d6_questions:
    page = q['gold_page']
    raw_quote = q['evidence_quote']
    page_chunks = chunks_by_page.get(page, [])
    
    matched_ev = None
    for c in page_chunks:
        ev = find_verbatim_evidence(raw_quote, c['text'])
        if ev:
            matched_ev = ev
            break
            
    if not matched_ev:
        raise ValueError(f"Could not match quote for {q['id']} on page {page}")
        
    item = {
        'id': q['id'],
        'split': q['split'],
        'type': q.get('type', 'truc_tiep'),
        'question': q['question'],
        'expected_answer': q['expected_answer'],
        'gold': [{
            'source_id': 'bo-luat-lao-dong-2019',
            'page': page,
            'evidence': matched_ev
        }]
    }
    if q.get('filter'):
        item['filter'] = q['filter']
    new_questions.append(item)

# Check splits
dev_ans = [q for q in new_questions if q['split'] == 'dev']
test_ans = [q for q in new_questions if q['split'] == 'test']
print(f"Processed 20 answerable questions: {len(dev_ans)} dev, {len(test_ans)} test.")

# Write data/chunks.json
out_data_dir = Path('./day07_vin_ai/K5-L3A-Day07-2A202603565-DOANVANHIEP-Retrieval-Context-Engineering/data')
out_data_dir.mkdir(parents=True, exist_ok=True)

with open(out_data_dir / 'chunks.json', 'w', encoding='utf-8') as f:
    json.dump(chunks, f, ensure_ascii=False, indent=2)

# Write data/questions.json (used by baseline.py)
with open(out_data_dir / 'questions.json', 'w', encoding='utf-8') as f:
    json.dump(new_questions, f, ensure_ascii=False, indent=2)

# Build cases.json for Agent Evaluation (total 24 cases: 17 dev, 7 test)
cases = []
for q in new_questions:
    cases.append({
        'id': q['id'],
        'split': q['split'],
        'question': q['question'],
        'answerable': True,
        'expected_behavior': 'search_and_answer',
        'expected_answer': q['expected_answer']
    })

# Add 2 unanswerable dev cases
cases.append({
    'id': 'q_unans_dev_1',
    'split': 'dev',
    'question': 'Mức đóng bảo hiểm y tế hộ gia đình hiện nay theo Luật Bảo hiểm Y tế là bao nhiêu phần trăm mức lương cơ sở?',
    'answerable': False,
    'expected_behavior': 'search_and_abstain',
    'absence_reason': 'Nội dung thuộc Luật Bảo hiểm Y tế, không quy định trong Bộ luật Lao động 2019.'
})
cases.append({
    'id': 'q_unans_dev_2',
    'split': 'dev',
    'question': 'Người điều khiển xe máy đi vào đường cao tốc bị xử phạt vi phạm hành chính bao nhiêu tiền theo quy định pháp luật?',
    'answerable': False,
    'expected_behavior': 'search_and_abstain',
    'absence_reason': 'Nội dung thuộc quy định xử phạt vi phạm giao thông đường bộ, không có trong Bộ luật Lao động 2019.'
})

# Add 2 unanswerable test cases
cases.append({
    'id': 'q_unans_test_1',
    'split': 'test',
    'question': 'Mức thuế suất thuế thu nhập cá nhân đối với bậc thu nhập trên 80 triệu đồng/tháng theo biểu thuế lũy tiến từng phần là bao nhiêu?',
    'answerable': False,
    'expected_behavior': 'search_and_abstain',
    'absence_reason': 'Nội dung thuộc Luật Thuế Thu nhập Cá nhân, không thuộc phạm vi Bộ luật Lao động 2019.'
})
cases.append({
    'id': 'q_unans_test_2',
    'split': 'test',
    'question': 'Thời hạn tối đa tạm giam bị can để phục vụ công tác điều tra tội phạm ít nghiêm trọng theo Bộ luật Tố tụng Hình sự là bao lâu?',
    'answerable': False,
    'expected_behavior': 'search_and_abstain',
    'absence_reason': 'Nội dung thuộc Bộ luật Tố tụng Hình sự, không có trong Bộ luật Lao động 2019.'
})

with open(out_data_dir / 'cases.json', 'w', encoding='utf-8') as f:
    json.dump(cases, f, ensure_ascii=False, indent=2)

print(f"Saved {len(cases)} cases (Dev: {len([c for c in cases if c['split']=='dev'])}, Test: {len([c for c in cases if c['split']=='test'])})")
