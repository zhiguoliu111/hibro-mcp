<div align="center">

# hibro - Universal Intelligent Memory Assistant

**🌐 Language / 语言** 

[English](./README.md) | [中文](./README_CN.md)

> Hibro let any AI coding tool to remember everything about you and provide personalized intelligent services

## 🎯 What is this?

**hibro** is a powerful universal intelligent memory system that supports any AI development tool compatible with the MCP (Model Context Protocol). It seamlessly integrates with Claude Code, Cursor, Qoder, Trae, and other IDEs, enabling your AI assistant to:

- 📝 **Remember your preferences** - Code style, technology choices, work habits
- 🧠 **Understand your decisions** - Architecture design, tech stack selection, project planning
- 🔍 **Intelligently retrieve information** - Keyword search + semantic similarity search
- 💡 **Proactively provide suggestions** - Recommend best practices based on historical data
- 🔒 **Protect data security** - Encrypted storage, automatic backup, security monitoring

## 😫 What pain points does it solve?

### 1. **AI Assistant's "Amnesia"**

**Problem**:
- Every new conversation, Claude starts from scratch, not remembering your preferences
- Repeatedly answering the same questions, low efficiency
- Unable to leverage historical decision experience

**hibro Solution**:
```
You: "I usually develop with React + TypeScript"
hibro: ✅ Remembered your tech stack preference

[One week later, new conversation]
Claude: I see you prefer React + TypeScript, shall we continue with this tech stack?
```

### 2. **Project Context Loss**

**Problem**:
- When switching projects, Claude doesn't know the project background
- Repeatedly explaining project architecture and technology choices
- Unable to leverage project historical experience

**hibro Solution**:
```
You: How far has the project progressed?
hibro: 📋 Auto-scanning project and generating snapshot...

  📊 my-webapp Project Progress Report
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🎯 Project Overview:
  Project Name: my-webapp
  Project Type: web application
  Tech Stack: React, TypeScript, Node.js, PostgreSQL
  Programming Languages: TypeScript, JavaScript

  📈 Project Statistics:
  • Total Files: 245
  • Project Size: 12.5 MB
  • Dependencies: 48

  🎯 Current Task: Implementing user authentication
  📅 Project Phase: development
  📊 Progress: 65%

  📝 Recent Work Records:
  • 2 hours ago: Completed login page UI
  • 1 day ago: Configured JWT authentication
  • 3 days ago: Designed user database model
```

### 3. **Repeated Mistakes**

**Problem**:
- Forgetting previous errors and solutions
- Repeatedly stepping into the same pitfalls
- Unable to learn from history

**hibro Solution**:
```
You: Encountered CORS error
hibro: 💡 Detected you encountered similar issue 2 weeks ago
  Previous solution: Add cors middleware in Express
  Related code: app.use(cors({ origin: 'http://localhost:3000' }))
```

### 4. **Information Fragmentation**

**Problem**:
- Important decisions scattered everywhere
- Difficult to trace decision reasons
- Team knowledge cannot be shared

**hibro Solution**:
```
You: Why choose PostgreSQL over MongoDB?
hibro: 📖 According to records, you made this decision on 2026-01-15:
  Reasons:
  1. Need complex queries and transaction support
  2. Team familiar with SQL databases
  3. High data consistency requirements
```

### 5. **Multi-Dialog Information Isolation** ✨ **NEW**

**Problem**:
- Information between multiple Claude dialog boxes is not interconnected
- Preferences stored in dialog A are unknown to dialog B
- Need to repeatedly explain the same preferences and decisions

**hibro Solution**:
```
Dialog A:
You: Remember, Java static constants must use uppercase and underscore naming
hibro: ✅ Stored in memory system

Dialog B (seconds later):
You: What are the Java static constant naming conventions?
hibro: 📋 According to your coding preferences:
  • Java static constants must use uppercase letters and underscores
  • Examples: MAX_SIZE, DEFAULT_TIMEOUT, API_BASE_URL
  • Avoid camelCase like maxSize
```

## ✨ Core Features

### 🔍 Automatic Project Scanning (New ✨)

**Intelligent Project Information Recognition**:
```python
You: How far has the project progressed?
hibro auto-scan:
  ✓ Project type (web/api/mobile/desktop/library)
  ✓ Tech stack (20+ technologies auto-identified)
  ✓ Framework (React, Vue, FastAPI, Django, etc.)
  ✓ Programming languages (Python, TypeScript, Java, etc.)
  ✓ Project scale (file count, size, dependencies)
  ✓ Dependencies (package.json, requirements.txt)
  ✓ Directory structure (deep analysis)
```

**Automatic Caching Mechanism**:
```
First scan: Complete project analysis (< 2 seconds)
Query within 7 days: Use cache, second-level response
Over 7 days: Auto re-scan, keep information fresh
```

**Use Cases**:
```
Scenario 1 - New project understanding:
  You: How far has the project progressed?
  → Auto-scan and generate project snapshot

Scenario 2 - Progress tracking:
  You: Check project progress
  → Display current tasks, recent work, statistics

Scenario 3 - Status update:
  You: Update project status, currently working on login feature, 50% progress
  → Record status and update active tasks
```

### 🔄 Multi-Dialog Real-time Sync (New ✨)

**Core Mechanism**:
```python
# Based on 3-layer sync mechanism
1. Database file monitoring
   - Monitor SQLite file changes
   - Detect database writes within 1 second
   - Auto-refresh cache version

2. Event publish/subscribe
   - 11 event types (memory storage, preference changes, etc.)
   - Asynchronous event processing
   - Priority notification mechanism

3. Cache version tracking
   - Version number +1 for each change
   - Claude judges whether refresh needed by version number
```

**Real-time Sync Flow**:
```
Dialog A                Shared Database                 Dialog B
   │                       │                         │
   ├─ Store: Java const uppercase ──►│                         │
   │                       │                         │
   │                  Write to DB                    │
   │                       │                         │
   │                  File change event ─────────────────────►│
   │                       │                    Detect change
   │                       │                         │
   │                  Publish events:                │
   │                  MEMORY_STORED                  │
   │                  PREFERENCE_CHANGED             │
   │                  DATABASE_CHANGED               │
   │                       │                         │
   │                       │                    Cache version++
   │                       │                    Version: 5
   │                       │                         │
   ◄───────────────────────┴─────────────────────────┤
                                                Next query
                                                Detect version change
                                                Auto-get latest memory
```

**Usage Example**:
```
Dialog A (Workspace 1):
You: Remember, MySQL field naming uses lowercase
hibro: ✅ Stored (published 3 events)
  📢 MEMORY_STORED
  📢 PREFERENCE_CHANGED
  📢 DATABASE_CHANGED

Dialog B (Workspace 2) - seconds later:
You: What are MySQL field naming conventions?
hibro: [Detected cache version 4→5, auto-refresh]
  📋 According to your coding preferences:
  • MySQL field naming convention: All field names use lowercase letters
  • Examples: user_id, created_at, user_name
  • Avoid uppercase forms like UserId, CreatedAt
```

**Monitor Sync Status**:
```
You: Check database sync status
hibro:
  ✅ Database listener running normally
  📊 Statistics:
  • Listen status: Running
  • Database path: ~/.hibro/memories.db
  • Change detections: 15 times
  • Last change: 30 seconds ago
  • Cache version: 23

  📈 Event bus status:
  • Worker threads: 2
  • Published events: 45
  • Processed events: 45
  • Subscribers: 2
```

### 📚 Intelligent Memory Storage

```python
# Auto-extract and store
Mentioned in conversation: I decided to use FastAPI framework
hibro auto:
  ✓ Extract type: Technical decision
  ✓ Rate importance: 0.8 (high)
  ✓ Generate semantic vector
  ✓ Associate with current project
  ✓ Encrypted storage
```

### 🔍 Dual-mode Search

**1. Keyword Search** - Exact matching
```
Search: "PostgreSQL configuration"
Result: Memories containing "PostgreSQL" and "configuration"
```

**2. Semantic Search** - Understanding meaning
```
Search: "database performance optimization"
Result: Even without these words, can find related:
  - Index optimization strategies
  - Query performance improvement
  - Cache configuration solutions
```

### 🧠 Deep Reasoning

**Causal Relationship Analysis**:
```
Detected causal chain:
  Choose microservice architecture → Need containerization → Choose Docker
  → Need orchestration tool → Choose Kubernetes
```

**Predictive Reasoning**:
```
Based on historical patterns predict:
  New FastAPI project → Likely need:
    ✓ SQLAlchemy (ORM)
    ✓ Pydantic (data validation)
    ✓ Alembic (database migration)
```

### 💡 Intelligent Guidance

**Tool Recommendations**:
```
Current task: Create new API endpoint
Recommended tools:
  1. analyze_project_deeply - Analyze existing API patterns
  2. build_knowledge_graph - View API dependencies
  3. get_smart_suggestions - Get best practice recommendations
```

**Learning Paths**:
```
Beginner users:
  Step 1: Basic memory storage
  Step 2: Simple search
  Step 3: Project context management
  ...
```

### 🔒 Security Assurance

- **Data Encryption**: AES-256-GCM encrypted storage
- **Automatic Backup**: Daily automatic backup with recovery support
- **Security Monitoring**: Real-time detection of abnormal access
- **Integrity Check**: Regular data integrity verification

## 🚀 Quick Start

### 1. Installation

```bash
# Clone repository
git clone <repository-url>
cd hibro

# Run installation script
python scripts/install.py
```

The installation script will automatically:
- ✅ Install all dependencies
- ✅ Configure MCP integration
- ✅ Create data directories
- ✅ Generate configuration files

### 2. Choose Running Mode

hibro supports two running modes, choose based on your use case:

#### Mode 1: stdio mode (Recommended for single IDE use)

```bash
# Direct startup (stdio mode)
python hibro.py
```

**IDE Configuration** (Any MCP-compatible IDE):
```json
{
  "mcpServers": {
    "hibro": {
      "command": "python",
      "args": ["D:/projects/hibro/hibro.py"]
    }
  }
}
```

#### Mode 2: Network daemon mode (Recommended for multiple IDEs)

```bash
# Start daemon
python hibro.py daemon start

# Check status
python hibro.py daemon status

# Stop daemon
python hibro.py daemon stop
```

**IDE Configuration** (Any MCP-compatible IDE):
```json
{
  "mcpServers": {
    "hibro": {
      "command": "tcp://localhost:8765"
    }
  }
}
```

### 3. Supported IDEs

hibro supports any MCP protocol-compatible IDE:

- ✅ **Claude Code** - Full support
- ✅ **Cursor** - Full support
- ✅ **Qoder** - Full support
- ✅ **Trae** - Full support
- ✅ **Any other MCP-compatible IDE** - Standard protocol support

### 4. Multi-IDE Usage Scenarios

#### Scenario 1: Single IDE use (stdio mode)
```bash
# Startup method
python hibro.py stdio
```
- ✅ Simple and direct, no additional configuration needed
- ✅ Fully compatible with existing IDE settings
- ✅ Minimal resource usage

#### Scenario 2: Multiple IDEs simultaneously (network mode)
```bash
# Start daemon
python hibro.py daemon start
```
Then all IDEs configure to connect to `tcp://localhost:8765`

- ✅ Multiple IDEs share the same memory data
- ✅ Resource efficient, only one server instance
- ✅ Data consistency guarantee

#### Scenario 3: Mixed use
```bash
# Support both modes simultaneously
# IDE1 uses stdio mode
# IDE2, IDE3 connect to network daemon
```

## 🏗️ Universal Architecture Design

### Dual-mode Architecture

```
┌─────────────────────────────────────┐
│           stdio mode                │
│                                     │
│  IDE → stdin/stdout → MCP Server    │
│                    ↓                │
│              Memory Engine          │
└─────────────────────────────────────┘

┌─────────────────────────────────────┐
│       Network daemon mode           │
│                                     │
│  IDE1 ↘                            │
│  IDE2 → tcp://localhost:8765 →     │
│  IDE3 ↗      MCP Daemon            │
│                    ↓                │
│           Shared Memory Engine      │
└─────────────────────────────────────┘
```

### Core Advantages

1. **Truly Universal**: Any MCP-compatible IDE can use it
2. **Flexible Deployment**: Choose appropriate mode based on use case
3. **Full Compatibility**: No need to modify existing configurations
4. **Resource Efficient**: Avoid multi-instance resource waste
5. **Data Consistency**: Multiple IDEs share the same intelligent memory

## 📖 Usage Examples

### Scenario 1: Project Progress Query (New Feature ✨)

```
You: How far has the project progressed?

hibro auto-scan and return:
  📊 my-webapp Project Progress Report
  ━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

  🎯 Project Overview:
  Project Name: my-webapp
  Project Type: web application
  Tech Stack: React, TypeScript, Node.js
  Framework: React
  Programming Languages: TypeScript, JavaScript

  📈 Project Statistics:
  • Total Files: 245
  • Total Directories: 87
  • Project Size: 12.5 MB
  • Dependencies: 48

  🎯 Current Task: Implementing user authentication
  📊 Progress: 65%

  📝 Recent Work:
  • 2 hours ago: Completed login page UI
  • 1 day ago: Configured JWT authentication
  • 3 days ago: Designed user database model

  💡 Suggestions:
  • Continue OAuth2 integration
  • Add unit test coverage
```

### Scenario 2: Multi-IDE Collaboration (New Feature ✨)

```
# Start daemon
python hibro.py daemon start

# In Claude Code
You: Remember, Java static constants must use uppercase and underscores
hibro: ✅ Stored in shared memory system

# In Cursor (seconds later)
You: What are Java constant naming conventions?
hibro: 📋 According to your coding preferences:
  • Java static constants must use uppercase letters and underscores
  • Examples: MAX_SIZE, DEFAULT_TIMEOUT, API_BASE_URL
  • Avoid camelCase like maxSize

# All IDEs can see the same memory data!
```

### Scenario 3: Record Technical Decisions

```
You: I decided to use Redis for caching because I need high concurrency support

hibro auto-record:
  Type: Technical decision
  Content: Choose Redis as caching solution
  Reason: Need high concurrency support
  Project: my-webapp
  Importance: 0.8 (high)
  Time: 2026-02-20 10:30
```

### Scenario 4: Query Historical Decisions

```
You: Why did we choose PostgreSQL before?

hibro answer:
  According to records (2026-01-15), you chose PostgreSQL because:
  1. Need complex queries and transaction support ✓
  2. Team familiar with SQL databases ✓
  3. High data consistency requirements ✓

  Related memories: 3 items
  Related projects: 2 projects
```

### Scenario 5: Get Intelligent Suggestions

```
You: I want to start a new React project

hibro suggestions:
  Based on your historical experience, recommend:
  1. Use TypeScript (90% of your projects use it)
  2. Configure ESLint + Prettier (your standard config)
  3. Use React Router (your common routing solution)

  Recommended toolchain:
    - Vite (build tool)
    - Zustand (state management)
    - React Query (data fetching)
```

### Scenario 6: Project Switching

```
You: Switch to my-api project

hibro load context:
  📋 Project: my-api
  📁 Path: /path/to/my-api
  🔧 Tech Stack: FastAPI + SQLAlchemy + PostgreSQL
  📊 Statistics:
    - Total memories: 45 items
    - Technical decisions: 12 items
    - Key issues: 5 items
  🎯 Active task: Implement payment feature

  Recent work:
    - 2 days ago: Completed user authentication
    - 5 days ago: Configured database migration
    - 1 week ago: Designed API architecture
```

## 🎓 Learning Path

### Beginner Users

1. **Basic Memory** - Learn how to store and retrieve information
2. **Project Association** - Associate memories with specific projects
3. **Simple Search** - Use keywords to find information
4. **Context Understanding** - Understand the role of project context

### Intermediate Users

5. **Semantic Search** - Use natural language search
6. **Decision Tracking** - Record and query technical decisions
7. **Workflow Optimization** - Improve efficiency using historical experience
8. **Team Collaboration** - Share project knowledge and experience

### Advanced Users

9. **Deep Analysis** - Causal relationship analysis and predictive reasoning
10. **Knowledge Graph** - Build and understand knowledge networks
11. **Automated Workflows** - Create intelligent workflows
12. **System Management** - Security configuration and performance optimization

## 🏗️ System Architecture

### Three-layer Architecture

```
┌─────────────────────────────────────┐
│      Claude Code (MCP Client)       │  AI assistant you use
└──────────────┬──────────────────────┘
               │ MCP Protocol
               ▼
┌─────────────────────────────────────┐
│     hibro MCP Server Layer        │  50+ intelligent tools
│  • Memory Mgmt  • AI Analysis  • Guidance │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│      Core Engine Layer              │  Core engine
│  • Memory Engine  • AI Analysis    │
│  • Security       • Backup         │
└──────────────┬──────────────────────┘
               │
               ▼
┌─────────────────────────────────────┐
│        Storage Layer                │  Data storage
│  • SQLite Database  • File System  │
│  • Encrypted Storage • Auto Backup │
└─────────────────────────────────────┘
```

### Core Modules

| Module | Responsibility | File Count |
|--------|----------------|------------|
| **core** | Core engine, memory management | 4 |
| **intelligence** | AI analysis, reasoning engine | 11 |
| **guidance** | Intelligent guidance, recommendation system | 4 |
| **assistant** | Proactive suggestions, workflow automation | 4 |
| **security** | Encryption, security monitoring | 4 |
| **backup** | Data backup, recovery | 3 |
| **mcp** | MCP protocol implementation | 1 |
| **storage** | Data storage, file management | 4 |

## 🔧 Configuration

### Basic Configuration (`~/.hibro/config.yaml`)

```yaml
# Memory settings
memory:
  max_memories: 100000          # Maximum memories
  default_importance: 0.5       # Default importance
  auto_learn: true              # Auto learning

# Semantic search
semantic_search:
  model_name: "all-MiniLM-L6-v2"  # Semantic model
  default_similarity_threshold: 0.3
  enable_cache: true

# Security settings
security:
  encrypt_data: true            # Encrypt data
  session_timeout: 3600         # Session timeout (seconds)

# Backup settings
backup:
  enabled: true                 # Enable backup
  interval_hours: 24            # Backup interval
  max_backups: 10               # Retain backup count
  encrypt_backups: true         # Encrypt backups
```

### Environment Variables

```bash
# Data directory
export MYJAVIS_DATA_DIR=/path/to/data

# Encryption password
export MYJAVIS_PASSWORD=your_secure_password

# Log level
export MYJAVIS_LOG_LEVEL=DEBUG
```

## 📊 Performance Metrics

### System Capacity
- **Maximum memories**: 100,000+
- **Database size**: Supports 10GB+
- **Concurrent sessions**: 100+
- **Response time**: < 2 seconds

### Search Performance
- **Keyword search**: < 100ms
- **Semantic search**: < 500ms (first time)
- **Semantic search**: < 100ms (cached)
- **Recommendation generation**: < 200ms

### Storage Efficiency
- **Compression ratio**: 60-70%
- **Cache hit rate**: 85%+
- **Backup speed**: 50MB/s

## 🔐 Security Features

### Data Encryption
- **Algorithm**: AES-256-GCM
- **Key derivation**: PBKDF2 (100,000 iterations)
- **Encryption scope**: Database, backup, cache

### Access Control
- Session timeout mechanism
- Abnormal access detection
- Audit log recording

### Backup Assurance
- Automatic backup (daily)
- Integrity verification
- Encrypted backup storage
- Disaster recovery support

## 🆘 Troubleshooting

### Common Issues

**Q: MCP server startup failed?**
```
A: Run diagnostic script
python test_mcp_startup.py

Common causes:
1. Python version < 3.10 → Upgrade Python
2. Missing dependencies → pip install -r requirements.txt
3. Model download timeout → Wait for download or configure proxy
```

**Q: No search results?**
```
A: Adjust search parameters
- Lower min_similarity threshold (0.3 → 0.2)
- Increase limit count (10 → 20)
- Use more general keywords
```

**Q: What if data is lost?**
```
A: Restore from backup
1. View available backups: get_backup_statistics
2. Restore backup: restore_backup
3. Verify data integrity
```

**Q: Can multiple dialog boxes sync in real-time?** ✨ **NEW**
```
A: Yes! hibro supports 3-layer sync mechanism
1. Database file monitoring - Detect changes within 1 second
2. Event publish/subscribe - 11 event types
3. Cache version tracking - Auto-judge if refresh needed

Test method:
- Dialog A: Remember, Java constants use UPPER_CASE naming
- Dialog B: Immediately ask "Java constant naming conventions"
- Expected: Dialog B can see the latest stored conventions

Check status: Call get_sync_status and get_event_bus_status
```

### Getting Help

- 📖 [Complete Documentation](./DEVELOPER_GUIDE.md)
- 🔧 [Troubleshooting Guide](./troubleshooting.md)
- 💡 [Best Practices](./best-practices.md)
- 📝 [Usage Examples](./usage-examples.md)

## 🗺️ Roadmap

### Completed ✅

- **Stage 1**: Deep reasoning capability upgrade
  - Causal relationship analysis
  - Predictive reasoning
  - Knowledge graph construction

- **Stage 2**: Adaptive learning mechanism
  - User behavior analysis
  - Dynamic scoring adjustment
  - Personalized recommendations

- **Stage 3**: Intelligent assistant capabilities
  - Proactive suggestion system
  - Workflow automation
  - Intelligent reminders

- **Stage 4**: Security and monitoring
  - Data encryption
  - Health monitoring
  - Automatic backup

- **Stage 5**: User experience enhancement ✨
  - MCP tool refactoring (53+ tools)
  - Intelligent guidance system
  - Complete documentation system

- **Stage 6**: Intelligent project management ✨ **NEW**
  - Automatic project scanning (20+ tech stack identification)
  - Project snapshot caching (7-day validity)
  - Intelligent progress tracking
  - Active task management

### Planned 🚧

- **Intelligent trigger mechanism** - Auto-identify project-related queries
- **Multimodal memory** - Support images, code, documents
- **Team collaboration** - Multi-user, permission management
- **API service** - RESTful API
- **Distributed storage** - Support large-scale deployment
- **Plugin system** - Third-party extension support

## 📈 Statistics

- **Lines of code**: ~27,500 lines
- **Python files**: 75 files
- **Core modules**: 13 modules
- **MCP tools**: 56+ tools
- **Documentation pages**: 6,500+ lines
- **New features**:
  - Automatic project scanning (2026-02-20)
  - Multi-dialog real-time sync (2026-02-20)
  - Event publish/subscribe system (2026-02-20)

## 🤝 Contributing

We welcome all forms of contributions!

### How to Contribute

1. Fork the repository
2. Create feature branch (`git checkout -b feature/amazing-feature`)
3. Commit changes (`git commit -m 'feat: add new feature'`)
4. Push to branch (`git push origin feature/amazing-feature`)
5. Create Pull Request

### Development Guide

- Follow PEP 8 code standards
- Add type annotations
- Write unit tests
- Update documentation

See [Developer Guide](./DEVELOPER_GUIDE.md) for details

## 📄 License

This project is licensed under the MIT License - see [LICENSE](../LICENSE) file for details

## 🙏 Acknowledgments

- [Claude Code](https://claude.ai/) - Excellent AI assistant
- [MCP SDK](https://github.com/modelcontextprotocol) - Powerful protocol support
- [sentence-transformers](https://www.sbert.net/) - Semantic understanding capabilities
- All contributors and users

---

<div align="center">

<div align="center">

**🌐 Language / 语言**

[English](./README.md) | [中文](./README_CN.md)

---

**[⬆ Back to Top](#hibro---universal-intelligent-memory-assistant)**

Made with ❤️ by hibro Team

**Version**: 2.2.0 | **Last Updated**: 2026-02-20 | **New**: Multi-dialog Real-time Sync

</div>