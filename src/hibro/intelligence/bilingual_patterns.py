#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Bilingual Pattern Configuration
Provides Chinese and English patterns for memory extraction and analysis
"""

# Bilingual memory type patterns
BILINGUAL_TYPE_PATTERNS = {
    'preference': {
        'patterns': [
            # Chinese patterns
            r'我(喜欢|偏好|习惯|通常|一般)',
            r'我的(风格|习惯|偏好|做法|偏好是)',
            r'(倾向于|更愿意|更喜欢)',
            r'(我认为|我觉得).*(更好|比较好|合适)',
            r'(以后|今后|以后请).*(请|要|遵循|使用)',
            r'代码.*偏好',
            r'注释.*偏好',
            # English patterns
            r'i\s+(like|prefer|usually|typically|generally)',
            r'my\s+(style|habit|preference|approach)',
            r'(tend\s+to|would\s+rather|prefer)',
            r'(i\s+think|i\s+feel).*(better|suitable)',
            r'(from\s+now\s+on|in\s+the\s+future).*(please|should|use|follow)',
            r'code.*preference',
            r'comment.*preference',
        ],
        'keywords': ['喜欢', '偏好', '习惯', '风格', '倾向', '愿意', '遵循', '以后',
                      'like', 'prefer', 'habit', 'style', 'tend', 'willing', 'follow', 'future'],
        'base_importance': 0.8
    },
    'decision': {
        'patterns': [
            # Chinese patterns
            r'(决定|选择|采用|使用).*(技术|框架|库|方案)',
            r'(架构|设计|实现).*方案',
            r'(最终|最后|确定).*(选择|决定|采用)',
            r'技术选型',
            r'项目.*决定',
            r'决定.*作为',
            r'选用',
            # English patterns
            r'(decide|choose|adopt|use).*(technology|framework|library|solution)',
            r'(architecture|design|implementation).*solution',
            r'(finally|eventually|confirm).*(choice|decision|adoption)',
            r'technology\s+selection',
            r'project.*decision',
            r'decide.*as',
            r'choose\s+to\s+use',
        ],
        'keywords': ['决定', '选择', '采用', '架构', '技术选型', '方案', '选用', '作为',
                      'decide', 'choose', 'adopt', 'architecture', 'tech', 'solution', 'selection'],
        'base_importance': 0.9
    },
    'project': {
        'patterns': [
            # Chinese patterns
            r'项目.*需求',
            r'功能.*实现',
            r'模块.*设计',
            r'系统.*架构',
            r'开发.*计划',
            r'这个项目',
            # English patterns
            r'project.*requirement',
            r'feature.*implementation',
            r'module.*design',
            r'system.*architecture',
            r'development.*plan',
            r'this\s+project',
        ],
        'keywords': ['项目', '需求', '功能', '模块', '系统', '开发',
                      'project', 'requirement', 'feature', 'module', 'system', 'development'],
        'base_importance': 0.7
    },
    'important': {
        'patterns': [
            # Chinese patterns
            r'(重要|关键|核心|必须).*注意',
            r'(记住|牢记|注意)',
            r'(特别|尤其|格外).*(重要|关键)',
            r'(千万|一定要|务必)',
            r'必须使用',
            r'一定要',
            # English patterns
            r'(important|key|critical|essential|must).*note',
            r'(remember|keep\s+in\s+mind|note)',
            r'(especially|particularly|exceptionally).*(important|key)',
            r'(must|be\ssure\s+to|make\ssure)',
            r'must\s+use',
            r'definitely',
        ],
        'keywords': ['重要', '关键', '核心', '记住', '注意', '必须', '一定要',
                      'important', 'key', 'critical', 'remember', 'note', 'must', 'definitely'],
        'base_importance': 1.0
    },
    'learning': {
        'patterns': [
            # Chinese patterns
            r'学到了',
            r'理解了',
            r'掌握了',
            r'发现.*问题',
            r'解决.*方法',
            # English patterns
            r'(learned|figured\s+out|mastered)',
            r'(understood|got|grasped)',
            r'(discovered|found).*(problem|issue)',
            r'solve.*method',
            r'solution.*for',
        ],
        'keywords': ['学到', '理解', '掌握', '发现', '解决',
                      'learn', 'understand', 'master', 'discover', 'solve'],
        'base_importance': 0.6
    }
}

# Bilingual importance boosters
BILINGUAL_IMPORTANCE_BOOSTERS = {
    # Chinese boosters
    '非常': 0.2, '特别': 0.2, '极其': 0.3, '绝对': 0.3,
    '必须': 0.2, '一定': 0.2, '关键': 0.2, '核心': 0.2, '重要': 0.1,
    # English boosters
    'very': 0.2, 'especially': 0.2, 'extremely': 0.3, 'absolutely': 0.3,
    'must': 0.2, 'definitely': 0.2, 'critical': 0.2, 'core': 0.2, 'important': 0.1
}

# Bilingual category keywords
BILINGUAL_CATEGORY_KEYWORDS = {
    # Chinese categories
    '编程语言': ['python', 'javascript', 'java', 'go', 'rust', 'typescript'],
    '框架库': ['react', 'vue', 'django', 'flask', 'express', 'spring'],
    '数据库': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite'],
    '工具': ['git', 'docker', 'kubernetes', 'jenkins', 'vscode'],
    '架构': ['微服务', '单体', '分布式', '云原生', 'serverless'],
    '测试': ['单元测试', '集成测试', '端到端测试', 'tdd', 'bdd'],
    '性能': ['优化', '缓存', '并发', '异步', '负载均衡'],
    '安全': ['认证', '授权', '加密', '防护', '漏洞'],
    # English categories
    'programming languages': ['python', 'javascript', 'java', 'go', 'rust', 'typescript'],
    'frameworks': ['react', 'vue', 'django', 'flask', 'express', 'spring'],
    'databases': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite'],
    'tools': ['git', 'docker', 'kubernetes', 'jenkins', 'vscode'],
    'architecture': ['microservices', 'monolithic', 'distributed', 'cloud-native', 'serverless'],
    'testing': ['unit test', 'integration test', 'e2e test', 'tdd', 'bdd'],
    'performance': ['optimization', 'cache', 'concurrency', 'async', 'load balancing'],
    'security': ['authentication', 'authorization', 'encryption', 'protection', 'vulnerabilities']
}

# Bilingual causal patterns
BILINGUAL_CAUSAL_PATTERNS = {
    'explicit': [
        # Chinese patterns
        (r'因为(.+?)所以(.+)', 'Because...so/therefore...'),
        (r'由于(.+?)导致(.+)', 'Due to...leads to...'),
        (r'(.+?)导致了?(.+)', '...caused...'),
        (r'(.+?)造成了?(.+)', '...resulted in...'),
        (r'(.+?)引起了?(.+)', '...triggered...'),
        (r'(.+?)的结果是(.+)', 'The result of...is...'),
        (r'基于(.+?)决定(.+)', 'Decided...based on...'),
        (r'考虑到(.+?)选择(.+)', 'Chose...considering...'),
        # English patterns
        (r'because(.+?)so(.+?)(?:,|\.)', 'Because...so...'),
        (r'due\s+to(.+?)lead(?:s|ing)\s+to(.+?)(?:,|\.)', 'Due to...leads to...'),
        (r'(.+?)caused(.+?)(?:,|\.)', '...caused...'),
        (r'(.+?)resulted\s+in(.+?)(?:,|\.)', '...resulted in...'),
        (r'(.+?)triggered(.+?)(?:,|\.)', '...triggered...'),
        (r'the\s+result\s+of(.+?)is(.+?)(?:,|\.)', 'The result of...is...'),
        (r'decided\s+to(.+?)based\s+on(.+?)(?:,|\.)', 'Decided to...based on...'),
        (r'chose(.+?)considering(.+?)(?:,|\.)', 'Chose...considering...'),
    ],
    'decision': [
        # Chinese patterns
        (r'决定使用(.+?)因为(.+)', 'Decided to use...because...'),
        (r'选择(.+?)是因为(.+)', 'Chose...because...'),
        (r'采用(.+?)的原因是(.+)', 'The reason for adopting...is...'),
        (r'之所以(.+?)是因为(.+)', 'The reason why...is...'),
        # English patterns
        (r'decided\s+to\s+use(.+?)because(.+?)(?:,|\.)', 'Decided to use...because...'),
        (r'chose(.+?)because(.+?)(?:,|\.)', 'Chose...because...'),
        (r'the\s+reason\s+for\s+adopting(.+?)is(.+?)(?:,|\.)', 'The reason for adopting...is...'),
        (r'the\s+reason\s+why(.+?)is(.+?)(?:,|\.)', 'The reason why...is...'),
    ],
    'temporal': [
        # Chinese patterns
        (r'首先(.+?)然后(.+)', 'First...then...'),
        (r'在(.+?)之后(.+)', 'After...do...'),
        (r'完成(.+?)后(.+)', 'After completing...do...'),
        (r'(.+?)接下来(.+)', '...next...'),
        # English patterns
        (r'first(.+?)then(.+?)(?:,|\.)', 'First...then...'),
        (r'after(.+?)(?:,|\.)?(?:then|do)(.+?)(?:,|\.)', 'After...do...'),
        (r'after\s+completing(.+?)(?:,|\.)?(?:then|do)(.+?)(?:,|\.)', 'After completing...do...'),
        (r'(.+?)next(.+?)(?:,|\.)', '...next...'),
    ]
}

# Bilingual stop words
BILINGUAL_STOP_WORDS = {
    # Chinese stop words
    '的', '了', '在', '是', '我', '有', '和', '就', '不', '人', '都', '一', '一个',
    '上', '也', '很', '到', '说', '要', '去', '你', '会', '着', '没有', '看', '好', '自己', '这',
    # English stop words
    'the', 'a', 'an', 'and', 'or', 'but', 'in', 'on', 'at', 'to', 'for', 'of',
    'with', 'by', 'from', 'as', 'is', 'was', 'are', 'been', 'be', 'have', 'has', 'had'
}

# Bilingual concept extraction patterns
BILINGUAL_CONCEPT_PATTERNS = [
    # Chinese patterns
    (r'[\u4e00-\u9fff]{2,4}(?:系统|平台|框架|工具|方法|模式|架构)', 'Chinese technical concepts'),
    (r'[\u4e00-\u9fff]{2,6}(?:管理|控制|分析|处理|优化)', 'Chinese action concepts'),
    (r'(?:基于|使用|采用|选择)[\u4e00-\u9fff]{2,6}', 'Chinese prefix-based concepts'),
    # English patterns
    (r'[A-Za-z]+(?:\s+(?:system|platform|framework|tool|method|pattern|architecture))', 'English technical concepts'),
    (r'[A-Za-z]+(?:\s+(?:management|control|analysis|processing|optimization))', 'English action concepts'),
    (r'(?:based\s+on|using|adopting|choosing)\s+[A-Za-z]+', 'English prefix-based concepts'),
]

# Bilingual role detection patterns
BILINGUAL_ROLE_PATTERNS = {
    'user': [
        # Chinese patterns
        (r'^(?:user|human|用户)[:：]', 'User role markers'),
        # English patterns
        (r'^(?:user|human|assistant|ai|claude)[:：]', 'User role markers (English)'),
    ],
    'assistant': [
        # Chinese patterns
        (r'^(?:assistant|ai|claude|助手|AI)[:：]', 'Assistant role markers'),
        # English patterns
        (r'^(?:assistant|ai|claude)[:：]', 'Assistant role markers (English)'),
    ]
}

# Bilingual decision keywords
BILINGUAL_DECISION_KEYWORDS = ['决策', 'decision', '选择', 'choice', '决定', 'decide']
