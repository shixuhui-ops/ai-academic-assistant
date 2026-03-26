import streamlit as st
import PyPDF2
import io
import os
from dotenv import load_dotenv
from openai import OpenAI
from rag_engine import RAGEngine
import networkx as nx
from pyvis.network import Network
import tempfile
import re
from collections import Counter

# 加载环境变量
load_dotenv()

st.set_page_config(page_title="AI学术助手", layout="wide")

st.title("📚 AI学术助手")
st.caption("智能论文分析 + 审稿模拟 + 文献增强")

# 初始化 OpenAI 客户端
client = OpenAI(
    api_key=os.getenv("DEEPSEEK_API_KEY"),
    base_url="https://api.deepseek.com"
)

# 初始化 RAG 引擎
if "rag_engine" not in st.session_state:
    st.session_state.rag_engine = RAGEngine()

# 侧边栏
with st.sidebar:
    st.markdown("### 📁 上传论文")
    
    # 单篇上传
    single_file = st.file_uploader("单篇上传（解读/审稿）", type=["pdf"], key="single")
    
    st.markdown("---")
    
    # 多篇上传
    st.markdown("### 📚 多篇上传（跨论文问答）")
    multi_files = st.file_uploader("支持多篇PDF", type=["pdf"], accept_multiple_files=True, key="multi")
    
    if multi_files:
        if st.button("📥 添加到知识库"):
            with st.spinner("正在处理论文..."):
                for file in multi_files:
                    pdf_reader = PyPDF2.PdfReader(io.BytesIO(file.read()))
                    full_text = ""
                    for page in pdf_reader.pages:
                        text = page.extract_text()
                        if text:
                            full_text += text
                    
                    st.session_state.rag_engine.add_paper(full_text, file.name)
                
                st.success(f"✅ 已添加 {len(multi_files)} 篇论文到知识库")
                st.rerun()
    
    # 显示已上传的论文列表
    current_papers = st.session_state.rag_engine.get_paper_list()
    if current_papers:
        st.markdown("---")
        st.markdown("### 📖 知识库论文")
        for paper in current_papers:
            st.write(f"- {paper}")
        
        if st.button("🗑️ 清空知识库"):
            st.session_state.rag_engine.clear()
            st.success("知识库已清空")
            st.rerun()

# 主界面 - 选项卡（6个）
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs(["📄 论文解读", "✍️ 审稿模拟", "💬 跨论文问答", "✏️ 润色翻译", "🕸️ 论文图谱", "📊 PPT大纲生成"])

# ========== Tab 1: 论文解读 ==========
with tab1:
    if single_file is not None:
        st.write(f"**当前论文：** {single_file.name}")
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(single_file.read()))
        full_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        
        if st.button("🔍 AI智能解读论文"):
            with st.spinner("AI正在分析论文中..."):
                prompt = f"""
                请分析以下论文，并按照JSON格式输出：
                1. 论文标题
                2. 摘要总结（100字以内）
                3. 研究方法
                4. 主要结果
                5. 创新点（3个点）
                
                论文内容：
                {full_text[:8000]}
                """
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一个专业的学术论文分析专家。"},
                        {"role": "user", "content": prompt}
                    ],
                    temperature=0.3
                )
                
                st.markdown("### 📊 论文解读结果")
                st.markdown(response.choices[0].message.content)
    else:
        st.info("👈 请在左侧侧边栏上传单篇论文")

# ========== Tab 2: 审稿模拟 ==========
with tab2:
    if single_file is not None:
        st.write(f"**当前论文：** {single_file.name}")
        
        pdf_reader = PyPDF2.PdfReader(io.BytesIO(single_file.getvalue()))
        full_text = ""
        for page in pdf_reader.pages:
            text = page.extract_text()
            if text:
                full_text += text
        
        if st.button("✍️ 模拟审稿"):
            with st.spinner("AI正在扮演审稿人..."):
                review_prompt = f"""
                请扮演一位顶会审稿人，对以下论文给出评审意见。
                
                评审维度：
                1. 创新性（1-10分）
                2. 实验设计（1-10分）
                3. 写作质量（1-10分）
                4. 整体评分（1-10分）
                
                请给出：
                - 优点（3点）
                - 缺点/不足（3点）
                - 修改建议（3条）
                - 是否建议接收
                
                论文内容：
                {full_text[:8000]}
                """
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一位资深学术审稿人，评审严格但公正。"},
                        {"role": "user", "content": review_prompt}
                    ],
                    temperature=0.5
                )
                
                st.markdown("### 📝 审稿意见")
                st.markdown(response.choices[0].message.content)
    else:
        st.info("👈 请在左侧侧边栏上传单篇论文")

# ========== Tab 3: 跨论文问答 ==========
with tab3:
    current_papers = st.session_state.rag_engine.get_paper_list()
    
    if len(current_papers) == 0:
        st.info("📚 请在左侧侧边栏上传多篇论文到知识库（至少2篇），然后进行跨论文问答")
    else:
        st.success(f"✅ 知识库中有 {len(current_papers)} 篇论文")
        
        with st.expander("查看知识库论文列表"):
            for paper in current_papers:
                st.write(f"- {paper}")
        
        # 调试信息
        with st.expander("🔧 调试：查看存储的论文内容"):
            previews = st.session_state.rag_engine.get_paper_texts_preview()
            if previews:
                for filename, preview in previews.items():
                    st.write(f"**{filename}**")
                    st.text(preview[:300])
                    st.markdown("---")
            else:
                st.write("暂无论文内容")
        
        question = st.text_input("💬 输入你的问题", placeholder="例如：对比这两篇论文的研究方法有什么不同？")
        
        if question and st.button("发送"):
            with st.spinner("正在检索并回答..."):
                context, paper_count = st.session_state.rag_engine.query(question)
                
                if context is None:
                    st.error("未找到相关内容，请尝试换个问题或上传更多论文")
                else:
                    answer_prompt = f"""
                    基于以下论文片段回答用户的问题。
                    
                    用户问题：{question}
                    
                    相关论文片段：
                    {context}
                    
                    请基于上述内容回答问题。如果信息不足以回答，请说明。回答要清晰、有条理。
                    """
                    
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一个学术助手，基于提供的论文内容回答问题。"},
                            {"role": "user", "content": answer_prompt}
                        ],
                        temperature=0.3
                    )
                    
                    st.markdown("### 💡 回答")
                    st.markdown(response.choices[0].message.content)
                    
                    with st.expander("📖 查看相关论文片段"):
                        st.markdown(context)

# ========== Tab 4: 润色翻译 ==========
with tab4:
    st.markdown("### 文本润色与翻译")
    
    # 输入文本区域
    input_text = st.text_area("输入需要处理的文本", height=200, placeholder="粘贴论文段落或摘要...")
    
    # 功能选择
    func_type = st.radio("选择功能", ["润色", "翻译"], horizontal=True)
    
    if func_type == "润色":
        style = st.selectbox("润色风格", ["academic", "critical", "concise"], 
                            format_func=lambda x: {"academic": "学术严谨", "critical": "批判性", "concise": "简洁"}[x])
        
        if st.button("✍️ 开始润色") and input_text:
            with st.spinner("润色中..."):
                prompts = {
                    "academic": "请将以下文本润色为学术论文风格，使用正式、严谨的语言，保持原意不变：",
                    "critical": "请以审稿人视角，用批判性学术语言改写以下文本，指出潜在问题或局限性，语气专业但尖锐：",
                    "concise": "请将以下文本精简，去除冗余表达，保留核心信息，保持学术风格："
                }
                
                prompt = prompts.get(style, prompts["academic"])
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一个专业的学术编辑助手。"},
                        {"role": "user", "content": f"{prompt}\n\n{input_text}"}
                    ],
                    temperature=0.7
                )
                
                st.markdown("### ✅ 润色结果")
                st.markdown(response.choices[0].message.content)
    
    else:  # 翻译
        direction = st.selectbox("翻译方向", ["zh→en", "en→zh"])
        
        if st.button("🌐 开始翻译") and input_text:
            with st.spinner("翻译中..."):
                if direction == "zh→en":
                    prompt = "请将以下中文内容翻译为英文，保持学术专业风格："
                else:
                    prompt = "请将以下英文内容翻译为中文，保持学术专业风格："
                
                response = client.chat.completions.create(
                    model="deepseek-chat",
                    messages=[
                        {"role": "system", "content": "你是一个专业的学术翻译助手。"},
                        {"role": "user", "content": f"{prompt}\n\n{input_text}"}
                    ],
                    temperature=0.3
                )
                
                st.markdown("### ✅ 翻译结果")
                st.markdown(response.choices[0].message.content)

# ========== Tab 5: 论文图谱可视化 ==========
with tab5:
    st.markdown("### 🕸️ 论文知识图谱")
    st.markdown("基于关键词共现关系，生成论文知识图谱")
    
    current_papers = st.session_state.rag_engine.get_paper_list()
    
    if len(current_papers) == 0:
        st.info("📚 请先在左侧侧边栏上传论文到知识库（至少1篇），然后生成图谱")
    else:
        st.success(f"✅ 知识库中有 {len(current_papers)} 篇论文")
        
        # 选择要生成图谱的论文
        selected_papers = st.multiselect(
            "选择要生成图谱的论文（默认全选）",
            options=current_papers,
            default=current_papers
        )
        
        # 选择关键词提取方式
        extract_method = st.radio(
            "关键词提取方式",
            ["AI智能提取（推荐）", "传统词频统计"],
            horizontal=True
        )
        
        # 高级选项
        with st.expander("⚙️ 图谱高级设置"):
            col1, col2 = st.columns(2)
            with col1:
                max_keywords = st.slider("最大关键词数量", 5, 25, 15)
                node_size_factor = st.slider("节点大小系数", 5, 30, 15)
            with col2:
                edge_threshold = st.slider("最小共现次数阈值", 0, 5, 1)
                gravity = st.slider("引力强度（越小越分散）", -2000, -100, -500)
        
        if st.button("🔍 生成知识图谱"):
            if not selected_papers:
                st.warning("请至少选择一篇论文")
            else:
                with st.spinner("正在提取关键词并生成图谱..."):
                    # 收集所有选中论文的文本
                    all_text = ""
                    for filename in selected_papers:
                        previews = st.session_state.rag_engine.get_paper_texts_preview()
                        if filename in previews:
                            all_text += previews[filename] + "\n\n"
                    
                    if len(all_text) < 100:
                        st.error("论文内容过少，请上传完整的论文PDF")
                        st.stop()
                    
                    # 根据选择的方式提取关键词
                    keywords = []
                    if extract_method == "AI智能提取（推荐）":
                        try:
                            kw_prompt = f"""
                            请从以下论文内容中提取{max_keywords}个核心关键词（学术概念、方法、技术、模型等）。
                            只返回关键词列表，每行一个，不要有其他内容。
                            
                            论文内容：
                            {all_text[:5000]}
                            """
                            
                            response = client.chat.completions.create(
                                model="deepseek-chat",
                                messages=[
                                    {"role": "system", "content": "你是一个学术关键词提取专家，只返回关键词列表。"},
                                    {"role": "user", "content": kw_prompt}
                                ],
                                temperature=0.3
                            )
                            
                            keywords_text = response.choices[0].message.content
                            keywords = [kw.strip() for kw in keywords_text.strip().split('\n') if kw.strip()]
                            keywords = keywords[:max_keywords]
                            
                            st.info(f"🤖 AI提取到 {len(keywords)} 个关键词")
                            
                        except Exception as e:
                            st.warning(f"AI提取失败，使用传统方法：{e}")
                            extract_method = "传统词频统计"
                    
                    if extract_method == "传统词频统计":
                        # 传统词频统计
                        words = re.findall(r'[\u4e00-\u9fa5a-zA-Z]{2,}', all_text)
                        stopwords = {'的', '了', '是', '在', '和', '与', '或', '对', '从', '这', '那', '什么', '怎么', '如何', '哪些', '哪个', '为什么', '一个', '这个', '那个', 'the', 'and', 'of', 'for', 'with', 'this', 'that', 'are', 'was', 'were', 'been', 'have', 'has', 'had', '论文', '提出', '方法', '实验', '结果', '分析', '研究', '进行', '使用', 'based', 'using', 'approach', 'model', 'data'}
                        filtered_words = [w.lower() for w in words if w.lower() not in stopwords and len(w) > 2]
                        word_freq = Counter(filtered_words)
                        top_keywords = word_freq.most_common(max_keywords)
                        keywords = [kw[0] for kw in top_keywords]
                        st.info(f"📊 词频统计提取到 {len(keywords)} 个关键词")
                    
                    if len(keywords) < 3:
                        st.warning("提取的关键词不足3个，请上传更完整的论文内容")
                        st.stop()
                    
                    # 计算关键词出现频率
                    kw_freq = {}
                    for kw in keywords:
                        # 使用正则匹配完整单词
                        pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                        count = len(re.findall(pattern, all_text.lower()))
                        kw_freq[kw] = max(count, 1)  # 至少为1
                    
                    # 计算关键词共现关系
                    # 将文本按句子分割
                    sentences = re.split(r'[。！？.!?]', all_text)
                    
                    # 初始化共现矩阵
                    co_occurrence = {}
                    for i in range(len(keywords)):
                        for j in range(i+1, len(keywords)):
                            co_occurrence[(keywords[i], keywords[j])] = 0
                    
                    # 计算共现
                    for sentence in sentences:
                        sentence_lower = sentence.lower()
                        present_keywords = []
                        for kw in keywords:
                            pattern = r'\b' + re.escape(kw.lower()) + r'\b'
                            if re.search(pattern, sentence_lower):
                                present_keywords.append(kw)
                        
                        for i in range(len(present_keywords)):
                            for j in range(i+1, len(present_keywords)):
                                kw1, kw2 = sorted([present_keywords[i], present_keywords[j]])
                                if (kw1, kw2) in co_occurrence:
                                    co_occurrence[(kw1, kw2)] += 1
                    
                    # 创建网络图
                    G = nx.Graph()
                    
                    # 计算节点大小（对数缩放，避免差异过大）
                    max_freq = max(kw_freq.values()) if kw_freq else 1
                    for kw in keywords:
                        # 节点大小 = 基础大小 + 频率比例 * 系数
                        size_ratio = kw_freq[kw] / max_freq
                        node_size = 15 + size_ratio * node_size_factor * 2
                        G.add_node(kw, 
                                   title=f"{kw}\n出现次数: {kw_freq[kw]}",
                                   size=node_size,
                                   label=kw)
                    
                    # 添加边（带权重）
                    max_weight = 1
                    edge_list = []
                    for (kw1, kw2), weight in co_occurrence.items():
                        if weight >= edge_threshold:
                            edge_list.append((kw1, kw2, weight))
                            if weight > max_weight:
                                max_weight = weight
                    
                    for kw1, kw2, weight in edge_list:
                        # 边宽度 = 1 + (权重/最大权重) * 5
                        edge_width = 1 + (weight / max_weight) * 5 if max_weight > 0 else 1
                        G.add_edge(kw1, kw2, 
                                   weight=weight,
                                   title=f"共现次数: {weight}",
                                   width=edge_width)
                    
                    # 如果没有边，显示警告
                    if G.number_of_edges() == 0:
                        st.warning("关键词之间没有共现关系，尝试调整最小共现次数阈值或上传更完整的论文")
                    else:
                        # 创建 PyVis 网络
                        net = Network(height="650px", 
                                      width="100%", 
                                      bgcolor="#f8f9fa", 
                                      font_color="black",
                                      notebook=False)
                        
                        net.from_nx(G)
                        
                        # 添加颜色分组（基于社区检测）
                        try:
                            # 使用 networkx 的社区检测
                            from networkx.algorithms import community
                            communities = community.greedy_modularity_communities(G)
                            
                            # 为每个社区分配颜色
                            color_palette = [
                                '#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd',
                                '#8c564b', '#e377c2', '#7f7f7f', '#bcbd22', '#17becf',
                                '#f4a261', '#e76f51', '#2a9d8f', '#e9c46a'
                            ]
                            
                            node_colors = {}
                            for idx, comm in enumerate(communities):
                                color = color_palette[idx % len(color_palette)]
                                for node in comm:
                                    node_colors[node] = color
                            
                            # 应用颜色到节点
                            for node in net.nodes:
                                if node['id'] in node_colors:
                                    node['color'] = node_colors[node['id']]
                                else:
                                    node['color'] = '#1f77b4'
                        except Exception as e:
                            # 如果社区检测失败，使用默认颜色
                            for node in net.nodes:
                                node['color'] = '#1f77b4'
                        
                        # 设置更优化的布局参数
                        net.set_options("""
                        {
                            "nodes": {
                                "font": { "size": 14, "face": "微软雅黑" },
                                "borderWidth": 2,
                                "borderWidthSelected": 3,
                                "shadow": {
                                    "enabled": true,
                                    "size": 5
                                }
                            },
                            "edges": {
                                "smooth": {
                                    "enabled": true,
                                    "type": "continuous"
                                },
                                "color": {
                                    "color": "#848484",
                                    "highlight": "#ff0000"
                                },
                                "selectionWidth": 2
                            },
                            "physics": {
                                "enabled": true,
                                "stabilization": {
                                    "enabled": true,
                                    "iterations": 200,
                                    "updateInterval": 25
                                },
                                "barnesHut": {
                                    "gravitationalConstant": -500,
                                    "centralGravity": 0.2,
                                    "springLength": 150,
                                    "springConstant": 0.05,
                                    "damping": 0.09,
                                    "avoidOverlap": 0.5
                                },
                                "maxVelocity": 50,
                                "minVelocity": 0.1,
                                "timestep": 0.5
                            },
                            "interaction": {
                                "hover": true,
                                "tooltipDelay": 200,
                                "navigationButtons": true,
                                "keyboard": true
                            }
                        }
                        """)
                        
                        # 保存为 HTML 文件
                        with tempfile.NamedTemporaryFile(delete=False, suffix=".html") as tmp_file:
                            net.save_graph(tmp_file.name)
                            tmp_file_path = tmp_file.name
                        
                        # 读取 HTML 并显示
                        with open(tmp_file_path, 'r', encoding='utf-8') as f:
                            html_content = f.read()
                        
                        # 添加自定义 CSS 让图谱更美观
                        html_content = html_content.replace(
                            '<style>',
                            '<style>#mynetwork { border-radius: 12px; box-shadow: 0 4px 12px rgba(0,0,0,0.1); } .vis-tooltip { background-color: rgba(0,0,0,0.8); color: white; border-radius: 6px; padding: 8px 12px; font-size: 12px; }</style>'
                        )
                        
                        st.components.v1.html(html_content, height=700)
                        
                        # 显示统计信息
                        st.markdown("---")
                        col1, col2, col3 = st.columns(3)
                        with col1:
                            st.metric("📊 关键词数量", len(keywords))
                        with col2:
                            st.metric("🔗 关联关系数", G.number_of_edges())
                        with col3:
                            avg_weight = sum(w for _, _, w in edge_list) / len(edge_list) if edge_list else 0
                            st.metric("⚡ 平均共现次数", f"{avg_weight:.1f}")
                        
                        # 显示关键词列表和共现关系
                        with st.expander("📊 详细数据：关键词及共现关系"):
                            col1, col2 = st.columns(2)
                            with col1:
                                st.markdown("**🔑 关键词列表（按频率排序）**")
                                sorted_keywords = sorted(kw_freq.items(), key=lambda x: x[1], reverse=True)
                                for kw, freq in sorted_keywords:
                                    st.write(f"- **{kw}** : {freq} 次")
                            
                            with col2:
                                st.markdown("**🔗 关键词共现关系（按强度排序）**")
                                edges_sorted = sorted(edge_list, key=lambda x: x[2], reverse=True)
                                for kw1, kw2, w in edges_sorted[:15]:
                                    # 显示进度条表示共现强度
                                    st.write(f"- {kw1} ↔ {kw2}")
                                    st.progress(min(w / max_weight, 1.0) if max_weight > 0 else 0, text=f"  共现 {w} 次")
                        
                        # 清理临时文件
                        os.unlink(tmp_file_path)

# ========== Tab 6: PPT大纲生成 ==========
with tab6:
    st.markdown("### 📊 PPT大纲生成")
    st.markdown("根据论文内容自动生成答辩/汇报PPT结构")
    
    # 获取知识库中的论文
    current_papers = st.session_state.rag_engine.get_paper_list()
    
    if len(current_papers) == 0:
        st.info("📚 请先在左侧侧边栏上传论文到知识库（至少1篇），然后生成PPT大纲")
    else:
        st.success(f"✅ 知识库中有 {len(current_papers)} 篇论文")
        
        # 选择要生成大纲的论文
        selected_paper = st.selectbox(
            "选择要生成PPT大纲的论文",
            options=current_papers
        )
        
        # 选择风格
        style = st.selectbox(
            "PPT风格",
            ["学术答辩", "开题报告", "组会汇报"],
            format_func=lambda x: {
                "学术答辩": "🎓 学术答辩（正式、完整）",
                "开题报告": "📖 开题报告（侧重研究背景和方案）", 
                "组会汇报": "👥 组会汇报（简洁、突出重点）"
            }[x]
        )
        
        # 是否包含演讲稿
        include_speech = st.checkbox("包含演讲稿提示（每页PPT的讲解要点）", value=True)
        
        if st.button("📊 生成PPT大纲"):
            if selected_paper:
                with st.spinner("AI正在生成PPT大纲..."):
                    # 获取论文内容
                    previews = st.session_state.rag_engine.get_paper_texts_preview()
                    paper_text = previews.get(selected_paper, "")
                    
                    if len(paper_text) < 500:
                        st.error("论文内容过少，请上传完整的论文PDF")
                        st.stop()
                    
                    # 根据风格设计不同的prompt
                    style_prompts = {
                        "学术答辩": """
                        请为这篇论文生成一个学术答辩PPT大纲。要求：
                        - 共10-12页PPT
                        - 包含：研究背景、相关工作、研究方法、实验设计、结果分析、创新点总结、未来工作、致谢
                        - 每页给出标题和3-5个要点
                        - 语言正式、学术化
                        """,
                        "开题报告": """
                        请为这篇论文生成一个开题报告PPT大纲。要求：
                        - 共8-10页PPT
                        - 包含：选题背景、研究意义、文献综述、研究目标、研究内容、技术路线、进度安排、预期成果
                        - 每页给出标题和3-5个要点
                        - 侧重研究方案的可行性和创新性
                        """,
                        "组会汇报": """
                        请为这篇论文生成一个组会汇报PPT大纲。要求：
                        - 共6-8页PPT
                        - 包含：背景介绍、核心方法、实验结果、遇到的困难、下一步计划
                        - 每页给出标题和2-4个要点
                        - 语言简洁，突出进展和问题
                        """
                    }
                    
                    speech_prompt = """
                    
                    此外，如果用户勾选了"包含演讲稿提示"，请在每页PPT后添加一段简短的演讲稿提示（50字以内），说明这一页应该如何讲解。
                    """ if include_speech else ""
                    
                    full_prompt = f"""
                    {style_prompts[style]}
                    {speech_prompt}
                    
                    论文内容：
                    {paper_text[:6000]}
                    
                    请直接输出PPT大纲，格式清晰，使用Markdown格式。每页PPT用"## 第X页"开头。
                    """
                    
                    response = client.chat.completions.create(
                        model="deepseek-chat",
                        messages=[
                            {"role": "system", "content": "你是一个学术PPT制作专家，擅长将论文内容转化为清晰的大纲。"},
                            {"role": "user", "content": full_prompt}
                        ],
                        temperature=0.5,
                        max_tokens=3000
                    )
                    
                    ppt_outline = response.choices[0].message.content
                    
                    # 显示结果
                    st.markdown("### 📋 PPT大纲")
                    st.markdown(ppt_outline)
                    
                    # 导出为Markdown文件
                    st.markdown("---")
                    
                    # 添加导出按钮
                    export_content = f"# {selected_paper} - PPT大纲\n\n"
                    export_content += f"**风格：** {style}\n\n"
                    export_content += f"**生成时间：** {st.session_state.get('current_time', '')}\n\n"
                    export_content += "---\n\n"
                    export_content += ppt_outline
                    
                    st.download_button(
                        label="📥 下载PPT大纲（Markdown格式）",
                        data=export_content,
                        file_name=f"{selected_paper.replace('.pdf', '')}_PPT大纲.md",
                        mime="text/markdown"
                    )
            else:
                st.warning("请先选择一篇论文")