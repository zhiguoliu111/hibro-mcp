#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Memory Extractor Module
Intelligently extracts important information from conversation content and classifies it
"""

import re
import logging
from typing import List, Dict, Any, Optional, Tuple
from datetime import datetime
from dataclasses import dataclass

from ..utils.helpers import sanitize_content
from .bilingual_patterns import BILINGUAL_TYPE_PATTERNS, BILINGUAL_IMPORTANCE_BOOSTERS, BILINGUAL_CATEGORY_KEYWORDS


@dataclass
class ExtractedMemory:
    """Extracted memory information"""
    content: str
    memory_type: str
    importance: float
    category: Optional[str] = None
    keywords: List[str] = None
    context: Optional[str] = None
    confidence: float = 0.5

    def __post_init__(self):
        if self.keywords is None:
            self.keywords = []


class MemoryExtractor:
    """Memory extractor"""

    def __init__(self):
        """Initialize memory extractor"""
        self.logger = logging.getLogger('hibro.memory_extractor')

        # Memory type recognition rules (Chinese text patterns for detection)
        self.type_patterns = {
            'preference': {
                'patterns': [
                    r'我(喜欢|偏好|习惯|通常|一般)',  # Matches: I like/prefer/am used to/usually/generally
                    r'我的(风格|习惯|偏好|做法|偏好是)',  # My (style/habit/preference/practice)
                    r'(倾向于|更愿意|更喜欢)',  # tend to/be more willing to/prefer
                    r'(我认为|我觉得).*(更好|比较好|合适)',  # (I think/I feel).*(better/suitable)
                    r'(以后|今后|以后请).*(请|要|遵循|使用)',  # (in future).*(please/must/follow/use)
                    r'代码.*偏好',  # Code.*preference
                    r'注释.*偏好',  # Comment.*preference
                ],
                'keywords': ['喜欢', '偏好', '习惯', '风格', '倾向', '愿意', '遵循', '以后'],
                'base_importance': 0.8
            },
            'decision': {
                'patterns': [
                    r'(决定|选择|采用|使用).*(技术|框架|库|方案)',  # (decide/choose/adopt/use).*(tech/framework/lib/solution)
                    r'(架构|设计|实现).*方案',  # (architecture/design/implementation).*solution
                    r'(最终|最后|确定).*(选择|决定|采用)',  # (finally/last/confirm).*(choice/decision/adoption)
                    r'技术选型',  # Technology selection
                    r'项目.*决定',  # Project.*decision
                    r'决定.*作为',  # Decide.*as
                    r'选用',  # Choose to use
                ],
                'keywords': ['决定', '选择', '采用', '架构', '技术选型', '方案', '选用', '作为'],
                'base_importance': 0.9
            },
            'project': {
                'patterns': [
                    r'项目.*需求',  # Project.*requirements
                    r'功能.*实现',  # Feature.*implementation
                    r'模块.*设计',  # Module.*design
                    r'系统.*架构',  # System.*architecture
                    r'开发.*计划',  # Development.*plan
                    r'这个项目',  # This project
                ],
                'keywords': ['项目', '需求', '功能', '模块', '系统', '开发'],
                'base_importance': 0.7
            },
            'important': {
                'patterns': [
                    r'(重要|关键|核心|必须).*注意',  # (important/key/core/must).*pay attention
                    r'(记住|牢记|注意)',  # (remember/keep in mind/pay attention)
                    r'(特别|尤其|格外).*(重要|关键)',  # (especially/particularly).*(important/key)
                    r'(千万|一定要|务必)',  # (must/be sure to/make sure to)
                    r'必须使用',  # Must use
                    r'一定要',  # Must definitely
                ],
                'keywords': ['重要', '关键', '核心', '记住', '注意', '必须', '一定要'],
                'base_importance': 1.0
            },
            'learning': {
                'patterns': [
                    r'学到了',  # Learned
                    r'理解了',  # Understood
                    r'掌握了',  # Mastered
                    r'发现.*问题',  # Discovered.*problem
                    r'解决.*方法'  # Solve.*method
                ],
                'keywords': ['学到', '理解', '掌握', '发现', '解决'],
                'base_importance': 0.6
            }
        }

        # Importance boosting keywords (Chinese intensity modifiers)
        self.importance_boosters = {
            '非常': 0.2,  # Very/Extremely
            '特别': 0.2,  # Especially
            '极其': 0.3,  # Extremely
            '绝对': 0.3,  # Absolutely
            '必须': 0.2,  # Must
            '一定': 0.2,  # Certainly/Definitely
            '关键': 0.2,  # Key/Critical
            '核心': 0.2,  # Core
            '重要': 0.1  # Important
        }

        # Category keywords (Chinese categories mapped to English terms)
        self.category_keywords = {
            '编程语言': ['python', 'javascript', 'java', 'go', 'rust', 'typescript'],  # Programming languages
            '框架库': ['react', 'vue', 'django', 'flask', 'express', 'spring'],  # Frameworks and libraries
            '数据库': ['mysql', 'postgresql', 'mongodb', 'redis', 'sqlite'],  # Databases
            '工具': ['git', 'docker', 'kubernetes', 'jenkins', 'vscode'],  # Tools
            '架构': ['微服务', '单体', '分布式', '云原生', 'serverless'],  # Architecture patterns
            '测试': ['单元测试', '集成测试', '端到端测试', 'tdd', 'bdd'],  # Testing types
            '性能': ['优化', '缓存', '并发', '异步', '负载均衡'],  # Performance topics
            '安全': ['认证', '授权', '加密', '防护', '漏洞']  # Security topics
        }

    def extract_memories(self, text: str, context: Optional[str] = None) -> List[ExtractedMemory]:
        """
        Extract memories from text

        Args:
            text: Input text
            context: Context information

        Returns:
            List of extracted memories
        """
        if not text or not text.strip():
            return []

        # Clean text
        cleaned_text = sanitize_content(text)

        # Split into sentences
        sentences = self._split_sentences(cleaned_text)

        memories = []
        for sentence in sentences:
            if len(sentence.strip()) < 10:  # Skip too short sentences
                continue

            memory = self._extract_from_sentence(sentence, context)
            if memory:
                memories.append(memory)

        # Merge similar memories
        memories = self._merge_similar_memories(memories)

        # Sort by importance
        memories.sort(key=lambda x: x.importance, reverse=True)

        self.logger.info(f"Extracted {len(memories)} memories from text")
        return memories

    def _split_sentences(self, text: str) -> List[str]:
        """
        Split text into sentences

        Args:
            text: Input text

        Returns:
            List of sentences
        """
        # Use regex to split sentences (Chinese and English punctuation)
        sentence_endings = r'[。！？；\n]+'  # Chinese and English sentence endings
        sentences = re.split(sentence_endings, text)

        # Clean empty and too short sentences
        sentences = [s.strip() for s in sentences if s.strip() and len(s.strip()) > 5]

        return sentences

    def _extract_from_sentence(self, sentence: str, context: Optional[str] = None) -> Optional[ExtractedMemory]:
        """
        Extract memory from a single sentence

        Args:
            sentence: Sentence text
            context: Context information

        Returns:
            Extracted memory, or None if no memory found
        """
        # Detect memory type and importance
        memory_type, base_importance, confidence = self._classify_sentence(sentence)

        if memory_type == 'conversation' and base_importance < 0.4:
            return None  # Skip unimportant general conversations

        # Calculate final importance
        importance = self._calculate_importance(sentence, base_importance)

        # Extract keywords
        keywords = self._extract_keywords(sentence)

        # Determine category
        category = self._determine_category(sentence, keywords)

        return ExtractedMemory(
            content=sentence,
            memory_type=memory_type,
            importance=min(importance, 1.0),  # Ensure not exceeding 1.0
            category=category,
            keywords=keywords,
            context=context,
            confidence=confidence
        )

    def _classify_sentence(self, sentence: str) -> Tuple[str, float, float]:
        """
        Classify sentence and calculate base importance

        Args:
            sentence: Sentence text

        Returns:
            (memory type, base importance, confidence)
        """
        sentence_lower = sentence.lower()

        best_type = 'conversation'
        best_importance = 0.3
        best_confidence = 0.5

        for memory_type, config in self.type_patterns.items():
            # Check pattern matches
            pattern_matches = 0
            for pattern in config['patterns']:
                try:
                    if re.search(pattern, sentence):
                        pattern_matches += 1
                except:
                    pass

            # Check keyword matches
            keyword_matches = 0
            for keyword in config['keywords']:
                if keyword in sentence or keyword in sentence_lower:
                    keyword_matches += 1

            # Calculate match score - any match will do
            if len(config['patterns']) > 0:
                pattern_score = pattern_matches / len(config['patterns'])
            else:
                pattern_score = 0

            if len(config['keywords']) > 0:
                keyword_score = keyword_matches / len(config['keywords'])
            else:
                keyword_score = 0

            # Modified: pattern match passes directly, keyword match needs threshold
            if pattern_matches > 0:
                total_score = 0.6  # Pattern match gets high score
            elif keyword_matches >= 2:
                total_score = 0.6  # 2+ keyword matches
            elif keyword_matches >= 1:
                total_score = 0.55  # 1 keyword match
            else:
                total_score = (pattern_score + keyword_score) / 2

            if total_score > best_confidence:
                best_type = memory_type
                best_importance = config['base_importance']
                best_confidence = total_score

        return best_type, best_importance, best_confidence

    def _calculate_importance(self, sentence: str, base_importance: float) -> float:
        """
        Calculate sentence importance score

        Args:
            sentence: Sentence text
            base_importance: Base importance

        Returns:
            Final importance score
        """
        importance = base_importance

        # Adjust based on importance boosters
        for booster, boost_value in self.importance_boosters.items():
            if booster in sentence:
                importance += boost_value

        # Adjust based on sentence length (moderate length sentences are usually more important)
        length = len(sentence)
        if 20 <= length <= 100:
            importance += 0.1
        elif length > 200:
            importance -= 0.1

        # Adjust based on punctuation
        if '!' in sentence:
            importance += 0.1
        if '？' in sentence or '?' in sentence:
            importance += 0.05

        return importance

    def _extract_keywords(self, sentence: str) -> List[str]:
        """
        Extract keywords from sentence

        Args:
            sentence: Sentence text

        Returns:
            List of keywords
        """
        keywords = []

        # Extract technical terms (English)
        tech_pattern = r'[A-Za-z][A-Za-z0-9]*(?:\.[A-Za-z][A-Za-z0-9]*)*'
        tech_words = re.findall(tech_pattern, sentence)
        keywords.extend([word.lower() for word in tech_words if len(word) > 2])

        # Extract Chinese keywords (simple word frequency based method)
        chinese_pattern = r'[\u4e00-\u9fff]+'  # Chinese character range
        chinese_words = re.findall(chinese_pattern, sentence)
        for word in chinese_words:
            if len(word) >= 2:
                keywords.append(word)

        # Deduplicate and limit count
        keywords = list(set(keywords))[:10]

        return keywords

    def _determine_category(self, sentence: str, keywords: List[str]) -> Optional[str]:
        """
        Determine sentence category

        Args:
            sentence: Sentence text
            keywords: List of keywords

        Returns:
            Category name, or None if unable to determine
        """
        sentence_lower = sentence.lower()
        keywords_lower = [kw.lower() for kw in keywords]

        category_scores = {}

        for category, category_keywords in self.category_keywords.items():
            score = 0
            for keyword in category_keywords:
                if keyword in sentence_lower:
                    score += 2
                if keyword in keywords_lower:
                    score += 1

            if score > 0:
                category_scores[category] = score

        if category_scores:
            return max(category_scores, key=category_scores.get)

        return None

    def _merge_similar_memories(self, memories: List[ExtractedMemory]) -> List[ExtractedMemory]:
        """
        Merge similar memories

        Args:
            memories: List of memories

        Returns:
            List of merged memories
        """
        if len(memories) <= 1:
            return memories

        merged = []
        used_indices = set()

        for i, memory1 in enumerate(memories):
            if i in used_indices:
                continue

            similar_memories = [memory1]
            used_indices.add(i)

            for j, memory2 in enumerate(memories[i+1:], i+1):
                if j in used_indices:
                    continue

                if self._are_similar_memories(memory1, memory2):
                    similar_memories.append(memory2)
                    used_indices.add(j)

            # If there are similar memories, merge them
            if len(similar_memories) > 1:
                merged_memory = self._merge_memory_group(similar_memories)
                merged.append(merged_memory)
            else:
                merged.append(memory1)

        return merged

    def _are_similar_memories(self, memory1: ExtractedMemory, memory2: ExtractedMemory) -> bool:
        """
        Determine if two memories are similar

        Args:
            memory1: Memory 1
            memory2: Memory 2

        Returns:
            Whether they are similar
        """
        # Types must match
        if memory1.memory_type != memory2.memory_type:
            return False

        # Categories must match
        if memory1.category != memory2.category:
            return False

        # Check keyword overlap
        common_keywords = set(memory1.keywords) & set(memory2.keywords)
        if len(common_keywords) >= 2:
            return True

        # Check content similarity (simple word overlap)
        words1 = set(memory1.content.split())
        words2 = set(memory2.content.split())
        overlap = len(words1 & words2) / len(words1 | words2)

        return overlap > 0.3

    def _merge_memory_group(self, memories: List[ExtractedMemory]) -> ExtractedMemory:
        """
        Merge a group of similar memories

        Args:
            memories: Group of memories

        Returns:
            Merged memory
        """
        # Select highest importance memory as main memory
        main_memory = max(memories, key=lambda m: m.importance)

        # Merge content
        contents = [m.content for m in memories]
        merged_content = ' | '.join(contents)

        # Merge keywords
        all_keywords = []
        for memory in memories:
            all_keywords.extend(memory.keywords)
        merged_keywords = list(set(all_keywords))

        # Calculate average importance
        avg_importance = sum(m.importance for m in memories) / len(memories)

        # Calculate average confidence
        avg_confidence = sum(m.confidence for m in memories) / len(memories)

        return ExtractedMemory(
            content=merged_content,
            memory_type=main_memory.memory_type,
            importance=avg_importance,
            category=main_memory.category,
            keywords=merged_keywords,
            context=main_memory.context,
            confidence=avg_confidence
        )