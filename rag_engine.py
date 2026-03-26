import hashlib
import json
import os
import re

class RAGEngine:
    def __init__(self, persist_file="./papers_cache.json"):
        self.persist_file = persist_file
        self.papers = []
        self.paper_texts = {}
        self._load_cache()
    
    def _load_cache(self):
        """加载缓存"""
        if os.path.exists(self.persist_file):
            try:
                with open(self.persist_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.papers = data.get("papers", [])
                    self.paper_texts = data.get("paper_texts", {})
            except:
                pass
    
    def _save_cache(self):
        """保存缓存"""
        with open(self.persist_file, 'w', encoding='utf-8') as f:
            json.dump({
                "papers": self.papers,
                "paper_texts": self.paper_texts
            }, f, ensure_ascii=False, indent=2)
    
    def add_paper(self, text, filename):
        """添加一篇论文"""
        paper_id = hashlib.md5(filename.encode()).hexdigest()
        
        # 存储论文（去重）
        if filename not in [p["filename"] for p in self.papers]:
            self.papers.append({
                "filename": filename,
                "paper_id": paper_id
            })
        
        # 存储论文内容
        self.paper_texts[filename] = text[:15000]
        
        self._save_cache()
        return True
    
    def query(self, question, top_k=3):
        """
        简单的关键词检索
        """
        if len(self.papers) == 0:
            return None, 0
        
        # 提取问题中的关键词
        keywords = self._extract_keywords(question)
        
        # 为每篇论文计算相关度
        scores = []
        for paper in self.papers:
            filename = paper["filename"]
            text = self.paper_texts.get(filename, "")
            
            if not text:
                continue
            
            # 简单相关度计算：关键词出现次数
            score = 0
            text_lower = text.lower()
            for kw in keywords:
                score += text_lower.count(kw.lower())
            
            # 也匹配论文名
            if filename.lower() in question.lower():
                score += 10
            
            scores.append((filename, score, text))
        
        # 按相关度排序
        scores.sort(key=lambda x: x[1], reverse=True)
        
        # 取 top_k 篇论文
        top_papers = [s for s in scores if s[1] > 0][:top_k]
        
        # 如果没有任何匹配，返回全部论文的前3000字
        if not top_papers:
            for paper in self.papers:
                filename = paper["filename"]
                text = self.paper_texts.get(filename, "")
                if text:
                    top_papers.append((filename, 0, text[:3000]))
        
        # 构建上下文
        context_parts = []
        for filename, score, text in top_papers:
            preview = text[:3000]
            context_parts.append(f"【{filename}】\n{preview}")
        
        if not context_parts:
            return None, len(self.papers)
        
        context_text = "\n\n---\n\n".join(context_parts)
        return context_text, len(self.papers)
    
    def _extract_keywords(self, question):
        """提取关键词"""
        stopwords = {"的", "了", "是", "在", "和", "与", "或", "对", "从", "这", "那", "什么", "怎么", "如何", "哪些", "哪个", "为什么", "一个", "这个", "那个"}
        
        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z0-9]+', question)
        
        keywords = []
        for w in words:
            if len(w) >= 2 and w not in stopwords:
                keywords.append(w)
        
        # 如果提取不到关键词，返回问题前10个字作为关键词
        if not keywords:
            keywords = [question[:10]]
        
        return keywords
    
    def get_paper_list(self):
        """获取已上传的论文列表"""
        return [p["filename"] for p in self.papers]
    
    def clear(self):
        """清空所有论文"""
        self.papers = []
        self.paper_texts = {}
        if os.path.exists(self.persist_file):
            os.remove(self.persist_file)
    
    def get_paper_texts_preview(self):
        """调试用：查看已存储的论文内容"""
        result = {}
        for filename, text in self.paper_texts.items():
            result[filename] = text[:500]
        return result