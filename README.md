# 🏗️ Fenestration Pro AI

**State-of-the-art AI assistant for shop drawings, specifications, and calculations**

Upload technical documents → Ask questions → Get expert answers with citations

---

## ✨ Features

### 🧠 Intelligent Document Understanding
- **Advanced OCR** with Google Document AI (optional)
- **Table extraction** from specs and BOMs
- **Dimension recognition** (ft, in, mm)
- **Material identification** (aluminum, glass, etc.)
- **Spec extraction** (U-factor, SHGC, STC, DP)
- **Calculation understanding** with formula parsing

### 🎯 Domain Expertise
- Window and door systems (casement, curtain wall, storefront)
- Glass specifications (Low-E, tempered, laminated, IGU)
- Performance metrics (U-factor, SHGC, VT, STC, DP)
- Materials (aluminum, vinyl, fiberglass, wood, steel)
- Standards (AAMA, ASTM, NFRC, ENERGY STAR)
- Technical details (anchor spacing, sill flashing, head details)

### 📚 Continuous Learning
- Learns from EVERY document upload (trainer and user)
- Recognizes patterns across drawings
- Builds relationships between documents
- Remembers successful query-document matches
- Gets smarter over time automatically

### 🎓 Grounded Answers
- ✅ **No hallucination** - answers only from your documents
- ✅ **Source citations** - every answer includes doc + page references
- ✅ **Confidence scores** - know when to trust the answer
- ✅ **Requests missing docs** - asks for specific drawings when needed

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
pip install -r requirements.txt
```

### 2. Set Up API Key

Create `.streamlit/secrets.toml`:

```toml
OPENAI_API_KEY = "sk-your-api-key-here"
```

### 3. Run the App

```bash
streamlit run app.py
```

### 4. Use the Interface

1. **Upload documents** - Drop PDFs or images of shop drawings
2. **Wait for processing** - AI extracts dimensions, materials, specs
3. **Ask questions** - "What are the window dimensions?" "What is the U-factor?"
4. **Get cited answers** - Every answer includes source references

---

## 🎯 Example Queries

**Dimensions:**
- "What are the dimensions of the main entrance door?"
- "Show me all window sizes"

**Materials:**
- "What material is specified for the curtain wall?"
- "List all glass types used"

**Specifications:**
- "What is the U-factor for the storefront system?"
- "What are the SHGC and VT ratings?"

**Calculations:**
- "Show me the wind load calculation"
- "What is the design pressure?"

**Details:**
- "What is the anchor spacing for the jambs?"
- "Show me the head detail requirements"

---

## 🔧 Advanced Features (Optional)

### Document AI (Google Cloud)

For best accuracy on scanned drawings:

1. Enable in sidebar: **Advanced Features → Enable Document AI**
2. Configure credentials:
   ```toml
   DOCAI_PROJECT_ID = "your-gcp-project"
   DOCAI_LOCATION = "us"
   DOCAI_PROCESSOR_ID = "your-processor-id"
   ```

See `INTELLIGENT_DOCUMENT_SETUP.md` for setup guide.

**Benefit:** 95%+ accuracy on scanned, handwritten, or poor-quality PDFs

### Pinecone (Vector Database)

For production-scale projects:

1. Enable in sidebar: **Advanced Features → Enable Pinecone**
2. Configure credentials:
   ```toml
   PINECONE_API_KEY = "your-pinecone-key"
   PINECONE_ENVIRONMENT = "us-east-1-aws"
   ```

**Benefit:** Faster search, metadata filtering, multi-project scaling

---

## 📊 How It Works

### Architecture

```
Upload Documents
    ↓
Document AI / pdfplumber (parsing)
    ↓
Entity Extraction (dimensions, materials, specs, tables)
    ↓
Intelligent Chunking (by sections, not tokens)
    ↓
OpenAI Embeddings
    ↓
Pinecone / ChromaDB (vector storage)
    ↓
User Query
    ↓
Hybrid Search (BM25 + Semantic)
    ↓
Re-ranking (top 8 results)
    ↓
GPT-4 Synthesis (with master system prompt)
    ↓
Structured Response (answer + citations + confidence)
```

### Master System Prompt

The AI operates in three modes:

1. **Admin Mode** - Document ingestion with coverage reports
2. **PM Mode** - Grounded Q&A with structured JSON responses
3. **Peer Review Mode** - Document validation against standards

All modes enforce:
- ✅ No hallucination (answers only from documents)
- ✅ Minimum 2 source citations
- ✅ Confidence scoring (0.0-1.0)
- ✅ Request missing artifacts when unsure
- ✅ Decline out-of-domain questions

### Continuous Learning

Every interaction teaches the AI:

- **Document uploads** → Pattern recognition (dimensions, materials, specs)
- **User queries** → Query-document success mapping
- **Relationships** → Similar docs, revisions, related systems
- **Knowledge graph** → Entity connections (systems → materials → specs)

Learning is persistent and automatic.

---

## 💰 Cost Estimates

### OpenAI (Required)
- **Embeddings**: ~$0.10 per 1M tokens
- **GPT-4o queries**: ~$0.005 per query
- **Typical cost**: $5-20/month for small firm

### Document AI (Optional)
- **First 1,000 pages/month**: FREE
- **After**: $1.50 per 1,000 pages
- **Typical cost**: $0-15/month

### Pinecone (Optional)
- **Serverless**: First 1M vectors FREE
- **Queries**: $0.045 per 1M queries
- **Typical cost**: $0-10/month

**Total: $5-45/month** for most fenestration firms

---

## 📁 Project Structure

```
fenestration-pro-ai/
├── app.py                          # Main Streamlit interface
├── config.py                       # Master system prompt & configs
├── domain_qa_engine.py             # Core QA engine (Admin/PM/Peer Review)
├── intelligent_document_parser.py  # Document AI + Pinecone integration
├── continuous_learning_engine.py   # Pattern recognition & learning
├── requirements.txt                # Python dependencies
├── INTELLIGENT_DOCUMENT_SETUP.md  # Advanced setup guide
├── README.md                       # This file
└── .streamlit/
    ├── config.toml                # UI theme
    └── secrets.toml               # API keys (create this)
```

---

## 🔒 Security & Privacy

- **Multi-tenant isolation** - No cross-company data leakage
- **Local-first option** - Can run fully offline with open-source stack
- **No training on your data** - OpenAI doesn't train on API calls
- **Encrypted storage** - All data encrypted at rest and in transit

---

## 🎓 Learning Statistics

View AI learning progress in real-time:

```
📚 Knowledge Base: 156 documents
📐 Unique Dimensions: 67
🔧 Unique Materials: 18
📋 Unique Specs: 12
🔍 Patterns Discovered: 145
🔗 Relationships Mapped: 89
💬 Queries Answered: 342
📈 Knowledge Nodes: 187
```

Export learning data for fine-tuning:

```python
engine.export_learning_for_finetuning("./training_data")
```

---

## 📖 Documentation

- `INTELLIGENT_DOCUMENT_SETUP.md` - Document AI & Pinecone setup
- `DEPLOYMENT_GUIDE.md` - Production deployment guide
- GitHub Issues - Bug reports and feature requests

---

## 🤝 Contributing

Contributions welcome! Please:

1. Fork the repository
2. Create a feature branch
3. Make your changes
4. Submit a pull request

---

## 📄 License

MIT License - see LICENSE file for details

---

## 🆘 Support

- **GitHub Issues**: https://github.com/Administrator2023/fenestration-pro-ai/issues
- **Documentation**: See `INTELLIGENT_DOCUMENT_SETUP.md`

---

## 🎯 Roadmap

- [ ] Image upload support (PNG, JPG) - In progress
- [ ] Drawing comparison (rev A vs rev B)
- [ ] Automated compliance checking
- [ ] Excel export of extracted data
- [ ] Mobile-responsive UI
- [ ] ERP integrations (Acumatica, QuickBooks)
- [ ] Custom entity training
- [ ] Red-lining and markup

---

**Built with ❤️ for fenestration professionals**
