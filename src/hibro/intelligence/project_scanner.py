#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Project Auto Scanner
Automatically scan project directories, extract key information, generate project snapshots
"""

import os
import json
import logging
from pathlib import Path
from typing import Dict, List, Any, Optional
from datetime import datetime
from dataclasses import dataclass, field


@dataclass
class ProjectSnapshot:
    """Project snapshot"""
    project_path: str
    scan_time: datetime
    project_name: str
    project_type: str  # web, api, mobile, desktop, library, etc.
    tech_stack: List[str]
    frameworks: List[str]
    languages: List[str]
    package_managers: List[str]
    key_files: Dict[str, str]  # filename -> content summary
    directory_structure: Dict[str, Any]
    statistics: Dict[str, int]
    dependencies: Dict[str, List[str]]
    metadata: Dict[str, Any] = field(default_factory=dict)


class ProjectScanner:
    """Project auto scanner"""

    # Project type identification rules
    PROJECT_TYPE_RULES = {
        'web': ['index.html', 'package.json', 'src/', 'public/'],
        'api': ['app.py', 'main.py', 'requirements.txt', 'api/'],
        'mobile': ['android/', 'ios/', 'react-native', 'flutter'],
        'desktop': ['electron', 'qt', '.desktop'],
        'library': ['setup.py', 'pyproject.toml', 'lib/'],
        'data_science': ['notebooks/', '.ipynb', 'data/'],
        'microservice': ['docker-compose.yml', 'Dockerfile', 'k8s/'],
    }

    # Tech stack identification rules
    TECH_STACK_RULES = {
        # Frontend frameworks
        'react': ['package.json', 'react'],
        'vue': ['package.json', 'vue'],
        'angular': ['package.json', '@angular'],
        'svelte': ['package.json', 'svelte'],

        # Backend frameworks
        'fastapi': ['requirements.txt', 'fastapi'],
        'django': ['requirements.txt', 'django'],
        'flask': ['requirements.txt', 'flask'],
        'express': ['package.json', 'express'],
        'spring': ['pom.xml', 'springframework'],

        # Databases
        'postgresql': ['requirements.txt', 'psycopg2', 'postgresql'],
        'mysql': ['requirements.txt', 'mysql', 'pymysql'],
        'mongodb': ['package.json', 'mongoose', 'requirements.txt', 'pymongo'],
        'redis': ['requirements.txt', 'redis', 'package.json', 'redis'],

        # Containerization
        'docker': ['Dockerfile', 'docker-compose.yml'],
        'kubernetes': ['k8s/', 'deployment.yaml', 'kubernetes'],

        # Others
        'typescript': ['tsconfig.json', 'package.json', 'typescript'],
        'python': ['requirements.txt', 'setup.py', 'pyproject.toml'],
        'nodejs': ['package.json'],
    }

    # Key files (need to read content)
    KEY_FILES = [
        'README.md',
        'readme.md',
        'package.json',
        'requirements.txt',
        'pyproject.toml',
        'setup.py',
        'Dockerfile',
        'docker-compose.yml',
        '.env.example',
        'config.yaml',
        'tsconfig.json',
    ]

    # Excluded directories
    EXCLUDE_DIRS = {
        'node_modules', '.git', '__pycache__', 'venv', '.venv',
        'dist', 'build', '.next', '.nuxt', 'coverage',
        '.pytest_cache', 'migrations', 'logs', 'cache'
    }

    # Excluded files
    EXCLUDE_FILES = {
        '.pyc', '.pyo', '.pyd', '.so', '.dll', '.dylib',
        '.log', '.tmp', '.bak', '.swp', '.swo'
    }

    def __init__(self):
        self.logger = logging.getLogger('hibro.project_scanner')

    def scan_project(self, project_path: str, quick_scan: bool = True) -> ProjectSnapshot:
        """
        Scan project directory

        Args:
            project_path: Project path
            quick_scan: Quick scan mode (only scan key files)

        Returns:
            Project snapshot
        """
        self.logger.info(f"Starting project scan: {project_path}")

        project_dir = Path(project_path)
        if not project_dir.exists():
            raise ValueError(f"Project path does not exist: {project_path}")

        # Identify project type
        project_type = self._identify_project_type(project_dir)

        # Identify tech stack
        tech_stack = self._identify_tech_stack(project_dir)

        # Identify frameworks
        frameworks = self._identify_frameworks(project_dir, tech_stack)

        # Identify programming languages
        languages = self._identify_languages(project_dir)

        # Identify package managers
        package_managers = self._identify_package_managers(project_dir)

        # Read key files
        key_files = self._read_key_files(project_dir)

        # Get directory structure
        directory_structure = self._get_directory_structure(
            project_dir,
            max_depth=2 if quick_scan else 4
        )

        # Statistics
        statistics = self._calculate_statistics(project_dir)

        # Extract dependencies
        dependencies = self._extract_dependencies(project_dir, key_files)

        # Project name
        project_name = self._extract_project_name(project_dir, key_files)

        snapshot = ProjectSnapshot(
            project_path=str(project_dir.absolute()),
            scan_time=datetime.now(),
            project_name=project_name,
            project_type=project_type,
            tech_stack=tech_stack,
            frameworks=frameworks,
            languages=languages,
            package_managers=package_managers,
            key_files=key_files,
            directory_structure=directory_structure,
            statistics=statistics,
            dependencies=dependencies,
            metadata={
                'scan_mode': 'quick' if quick_scan else 'full',
                'scanned_at': datetime.now().isoformat()
            }
        )

        self.logger.info(f"Project scan completed: {project_name} ({project_type})")
        return snapshot

    def _identify_project_type(self, project_dir: Path) -> str:
        """Identify project type"""
        scores = {}

        for project_type, indicators in self.PROJECT_TYPE_RULES.items():
            score = 0
            for indicator in indicators:
                if (project_dir / indicator).exists():
                    score += 1
            scores[project_type] = score

        # Return the type with highest score
        if not scores or max(scores.values()) == 0:
            return 'unknown'

        return max(scores, key=scores.get)

    def _identify_tech_stack(self, project_dir: Path) -> List[str]:
        """Identify tech stack"""
        tech_stack = []

        for tech, rules in self.TECH_STACK_RULES.items():
            for i in range(0, len(rules), 2):
                file_name = rules[i]
                keyword = rules[i + 1] if i + 1 < len(rules) else None

                file_path = project_dir / file_name
                if file_path.exists():
                    if keyword is None:
                        tech_stack.append(tech)
                        break
                    else:
                        # Check if file content contains keyword
                        try:
                            content = file_path.read_text(encoding='utf-8', errors='ignore')
                            if keyword in content.lower():
                                tech_stack.append(tech)
                                break
                        except Exception:
                            pass

        return list(set(tech_stack))

    def _identify_frameworks(self, project_dir: Path, tech_stack: List[str]) -> List[str]:
        """Identify frameworks"""
        frameworks = []

        # Extract frameworks from tech stack
        framework_keywords = {
            'react', 'vue', 'angular', 'svelte',
            'fastapi', 'django', 'flask', 'express', 'spring'
        }

        for tech in tech_stack:
            if tech in framework_keywords:
                frameworks.append(tech)

        return frameworks

    def _identify_languages(self, project_dir: Path) -> List[str]:
        """Identify programming languages"""
        languages = []

        # Count by file extensions
        extensions = {
            '.py': 'Python',
            '.js': 'JavaScript',
            '.ts': 'TypeScript',
            '.jsx': 'JavaScript (React)',
            '.tsx': 'TypeScript (React)',
            '.java': 'Java',
            '.go': 'Go',
            '.rs': 'Rust',
            '.cpp': 'C++',
            '.c': 'C',
            '.rb': 'Ruby',
            '.php': 'PHP',
            '.swift': 'Swift',
            '.kt': 'Kotlin',
        }

        file_counts = {}
        for file_path in project_dir.rglob('*'):
            if file_path.is_file():
                ext = file_path.suffix.lower()
                if ext in extensions:
                    lang = extensions[ext]
                    file_counts[lang] = file_counts.get(lang, 0) + 1

        # Return the top 3 languages with most files
        if file_counts:
            sorted_langs = sorted(file_counts.items(), key=lambda x: x[1], reverse=True)
            languages = [lang for lang, count in sorted_langs[:3]]

        return languages

    def _identify_package_managers(self, project_dir: Path) -> List[str]:
        """Identify package managers"""
        package_managers = []

        if (project_dir / 'package.json').exists():
            if (project_dir / 'yarn.lock').exists():
                package_managers.append('yarn')
            elif (project_dir / 'pnpm-lock.yaml').exists():
                package_managers.append('pnpm')
            else:
                package_managers.append('npm')

        if (project_dir / 'requirements.txt').exists():
            package_managers.append('pip')

        if (project_dir / 'Pipfile').exists():
            package_managers.append('pipenv')

        if (project_dir / 'pyproject.toml').exists():
            if (project_dir / 'poetry.lock').exists():
                package_managers.append('poetry')

        if (project_dir / 'pom.xml').exists():
            package_managers.append('maven')

        if (project_dir / 'build.gradle').exists():
            package_managers.append('gradle')

        return package_managers

    def _read_key_files(self, project_dir: Path) -> Dict[str, str]:
        """Read key file contents"""
        key_files = {}

        for file_name in self.KEY_FILES:
            file_path = project_dir / file_name
            if file_path.exists():
                try:
                    content = file_path.read_text(encoding='utf-8', errors='ignore')
                    # Only keep first 1000 characters as summary
                    key_files[file_name] = content[:1000]
                except Exception as e:
                    self.logger.warning(f"Cannot read file {file_name}: {e}")

        return key_files

    def _get_directory_structure(self, project_dir: Path, max_depth: int = 2) -> Dict[str, Any]:
        """Get directory structure"""
        def build_tree(path: Path, depth: int = 0) -> Dict[str, Any]:
            if depth > max_depth:
                return {}

            tree = {}
            try:
                for item in sorted(path.iterdir()):
                    # Exclude specific directories and files
                    if item.name in self.EXCLUDE_DIRS:
                        continue
                    if item.suffix in self.EXCLUDE_FILES:
                        continue
                    if item.name.startswith('.'):
                        continue

                    if item.is_dir():
                        tree[item.name] = {
                            'type': 'directory',
                            'children': build_tree(item, depth + 1)
                        }
                    else:
                        tree[item.name] = {
                            'type': 'file',
                            'size': item.stat().st_size
                        }
            except PermissionError:
                pass

            return tree

        return build_tree(project_dir)

    def _calculate_statistics(self, project_dir: Path) -> Dict[str, int]:
        """Calculate project statistics"""
        stats = {
            'total_files': 0,
            'total_dirs': 0,
            'total_size_mb': 0,
            'by_extension': {}
        }

        total_size = 0
        for item in project_dir.rglob('*'):
            # Exclude specific directories
            if any(excluded in item.parts for excluded in self.EXCLUDE_DIRS):
                continue

            if item.is_file():
                stats['total_files'] += 1
                total_size += item.stat().st_size

                # Count by extension
                ext = item.suffix or 'no_extension'
                stats['by_extension'][ext] = stats['by_extension'].get(ext, 0) + 1

            elif item.is_dir():
                stats['total_dirs'] += 1

        stats['total_size_mb'] = round(total_size / (1024 * 1024), 2)

        return stats

    def _extract_dependencies(self, project_dir: Path, key_files: Dict[str, str]) -> Dict[str, List[str]]:
        """Extract project dependencies"""
        dependencies = {
            'python': [],
            'node': [],
            'java': [],
            'other': []
        }

        # Python dependencies
        if 'requirements.txt' in key_files:
            try:
                content = key_files['requirements.txt']
                for line in content.split('\n'):
                    line = line.strip()
                    if line and not line.startswith('#'):
                        # Extract package name (remove version)
                        pkg = line.split('==')[0].split('>=')[0].split('<=')[0].split('~=')[0].strip()
                        if pkg:
                            dependencies['python'].append(pkg)
            except Exception as e:
                self.logger.warning(f"Failed to parse requirements.txt: {e}")

        # Node.js dependencies
        if 'package.json' in key_files:
            try:
                content = key_files['package.json']
                pkg_data = json.loads(content)
                for dep_type in ['dependencies', 'devDependencies']:
                    if dep_type in pkg_data:
                        dependencies['node'].extend(list(pkg_data[dep_type].keys()))
            except Exception as e:
                self.logger.warning(f"Failed to parse package.json: {e}")

        return dependencies

    def _extract_project_name(self, project_dir: Path, key_files: Dict[str, str]) -> str:
        """Extract project name"""
        # 1. Extract from package.json
        if 'package.json' in key_files:
            try:
                pkg_data = json.loads(key_files['package.json'])
                if 'name' in pkg_data:
                    return pkg_data['name']
            except Exception:
                pass

        # 2. Extract from pyproject.toml
        if 'pyproject.toml' in key_files:
            try:
                content = key_files['pyproject.toml']
                # Simple extraction of name field
                for line in content.split('\n'):
                    if 'name' in line and '=' in line:
                        return line.split('=')[1].strip().strip('"\'')
            except Exception:
                pass

        # 3. Use directory name
        return project_dir.name

    def generate_summary(self, snapshot: ProjectSnapshot) -> str:
        """
        Generate project summary text

        Args:
            snapshot: Project snapshot

        Returns:
            Project summary
        """
        summary_lines = [
            f"📁 Project Name: {snapshot.project_name}",
            f"📂 Project Path: {snapshot.project_path}",
            f"🏷️ Project Type: {snapshot.project_type}",
            "",
            "🔧 Tech Stack:",
        ]

        for tech in snapshot.tech_stack:
            summary_lines.append(f"  • {tech}")

        if snapshot.frameworks:
            summary_lines.append("")
            summary_lines.append("🖼️ Frameworks:")
            for framework in snapshot.frameworks:
                summary_lines.append(f"  • {framework}")

        if snapshot.languages:
            summary_lines.append("")
            summary_lines.append("💻 Programming Languages:")
            for lang in snapshot.languages:
                summary_lines.append(f"  • {lang}")

        summary_lines.extend([
            "",
            "📊 Project Statistics:",
            f"  • Total Files: {snapshot.statistics['total_files']:,}",
            f"  • Total Directories: {snapshot.statistics['total_dirs']:,}",
            f"  • Project Size: {snapshot.statistics['total_size_mb']} MB",
        ])

        if snapshot.dependencies:
            total_deps = sum(len(deps) for deps in snapshot.dependencies.values())
            summary_lines.append(f"  • Dependencies: {total_deps}")

        return '\n'.join(summary_lines)
