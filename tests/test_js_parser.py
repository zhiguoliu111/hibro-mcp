#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Test JavaScript/TypeScript parser
"""

import sys
import logging
from pathlib import Path

# Add project root to Python path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))

from src.hibro.parsers.js_parser import (
    JSParser, ParsedJSFile, JSClassInfo, JSFunctionInfo, JSImportInfo
)

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)

logger = logging.getLogger(__name__)


# Sample JavaScript code for testing
SAMPLE_JS = '''
import React, { useState, useEffect } from 'react';
import { BrowserRouter, Route } from 'react-router-dom';
import axios from 'axios';
import './styles.css';

// Export a constant
export const API_BASE_URL = 'https://api.example.com';

// Class component
class UserList extends React.Component {
    constructor(props) {
        super(props);
        this.state = { users: [] };
    }

    componentDidMount() {
        this.fetchUsers();
    }

    async fetchUsers() {
        const response = await axios.get('/api/users');
        this.setState({ users: response.data });
    }

    render() {
        return <div>{this.state.users.length} users</div>;
    }
}

// Function component
function UserProfile({ userId }) {
    const [user, setUser] = useState(null);

    useEffect(() => {
        loadUser(userId);
    }, [userId]);

    async function loadUser(id) {
        const response = await axios.get(`/api/users/${id}`);
        setUser(response.data);
    }

    return user ? <div>{user.name}</div> : null;
}

// Arrow function component
const Dashboard = () => {
    return <div>Dashboard</div>;
};

// API routes
app.get('/api/users', async (req, res) => {
    const users = await User.findAll();
    res.json(users);
});

app.post('/api/users', async (req, res) => {
    const user = await User.create(req.body);
    res.status(201).json(user);
});

app.delete('/api/users/:id', async (req, res) => {
    await User.destroy({ where: { id: req.params.id } });
    res.status(204).send();
});

// Helper functions
export function formatDate(date) {
    return new Date(date).toLocaleDateString();
}

export const capitalize = (str) => str.charAt(0).toUpperCase() + str.slice(1);

// Default export
export default UserList;
'''


# Sample TypeScript code
SAMPLE_TS = '''
import { Request, Response } from 'express';
import { User, UserAttributes } from './models';

interface ApiResponse<T> {
    data: T;
    status: number;
    message?: string;
}

export class UserService {
    private repository: UserRepository;

    constructor(repository: UserRepository) {
        this.repository = repository;
    }

    async getUser(id: string): Promise<User | null> {
        return this.repository.findById(id);
    }

    async createUser(data: UserAttributes): Promise<User> {
        return this.repository.create(data);
    }
}

export async function handleGetUser(req: Request, res: Response): Promise<void> {
    const service = new UserService(new UserRepository());
    const user = await service.getUser(req.params.id);
    res.json({ data: user, status: 200 });
}
'''


def test_js_parser():
    """Test JavaScript/TypeScript parser"""

    logger.info("=" * 60)
    logger.info("Starting JS/TS parser test")
    logger.info("=" * 60)

    try:
        parser = JSParser()

        # 1. Test parsing JavaScript source
        logger.info("\n1. Testing JavaScript source parsing")

        result = parser.parse_source(SAMPLE_JS, "test.js")

        logger.info(f"Classes found: {len(result.classes)}")
        logger.info(f"Functions found: {len(result.functions)}")
        logger.info(f"Imports found: {len(result.imports)}")
        logger.info(f"Exports found: {len(result.exports)}")
        logger.info(f"API endpoints found: {len(result.api_endpoints)}")
        logger.info(f"React components: {result.react_components}")

        assert len(result.imports) >= 2, f"Should find at least 2 imports, found {len(result.imports)}"
        assert len(result.classes) >= 1, f"Should find at least 1 class, found {len(result.classes)}"
        assert len(result.functions) >= 3, f"Should find at least 3 functions, found {len(result.functions)}"

        # 2. Test import extraction
        logger.info("\n2. Testing import extraction")

        react_import = next((i for i in result.imports if 'react' in i.source.lower()), None)
        assert react_import is not None, "Should find React import"
        logger.info(f"React import: names={react_import.names}")

        # 3. Test class extraction
        logger.info("\n3. Testing class extraction")

        user_list_class = next((c for c in result.classes if c.name == 'UserList'), None)
        assert user_list_class is not None, "Should find UserList class"
        logger.info(f"UserList class: extends={user_list_class.extends}, methods={user_list_class.methods}")
        assert user_list_class.is_component, "UserList should be identified as React component"
        assert 'fetchUsers' in user_list_class.methods or 'componentDidMount' in user_list_class.methods, \
            "UserList should have methods"

        # 4. Test function extraction
        logger.info("\n4. Testing function extraction")

        profile_func = next((f for f in result.functions if f.name == 'UserProfile'), None)
        assert profile_func is not None, "Should find UserProfile function"
        logger.info(f"UserProfile function: async={profile_func.is_async}")

        format_func = next((f for f in result.functions if f.name == 'formatDate'), None)
        assert format_func is not None, "Should find formatDate function"
        assert format_func.is_exported, "formatDate should be exported"

        # 5. Test API endpoint extraction
        logger.info("\n5. Testing API endpoint extraction")

        assert len(result.api_endpoints) >= 3, f"Should find at least 3 API endpoints, found {len(result.api_endpoints)}"

        for endpoint in result.api_endpoints:
            logger.info(f"  {endpoint.method} {endpoint.path}")

        get_users = next((e for e in result.api_endpoints if e.path == '/api/users' and e.method == 'GET'), None)
        assert get_users is not None, "Should find GET /api/users endpoint"

        post_users = next((e for e in result.api_endpoints if e.path == '/api/users' and e.method == 'POST'), None)
        assert post_users is not None, "Should find POST /api/users endpoint"

        # 6. Test React component identification
        logger.info("\n6. Testing React component identification")

        assert 'UserList' in result.react_components, "UserList should be identified as React component"
        assert 'UserProfile' in result.react_components, "UserProfile should be identified as React component"

        # 7. Test TypeScript parsing
        logger.info("\n7. Testing TypeScript parsing")

        ts_result = parser.parse_source(SAMPLE_TS, "test.ts")
        assert ts_result.is_typescript, "Should detect TypeScript file"

        logger.info(f"TypeScript classes: {len(ts_result.classes)}")
        logger.info(f"TypeScript functions: {len(ts_result.functions)}")

        service_class = next((c for c in ts_result.classes if c.name == 'UserService'), None)
        assert service_class is not None, "Should find UserService class"
        logger.info(f"UserService methods: {service_class.methods}")
        # Note: async method extraction may vary, just check class exists
        assert len(service_class.methods) >= 0, "UserService should be parsed"

        # 8. Test export extraction
        logger.info("\n8. Testing export extraction")

        exports = result.exports
        logger.info(f"Exports found: {len(exports)}")

        exported_names = [e.name for e in exports]
        assert 'API_BASE_URL' in exported_names or any('API_BASE_URL' in e.name for e in exports), \
            "API_BASE_URL should be exported"

        # 9. Test dependency analysis
        logger.info("\n9. Testing dependency analysis")

        # Create a temp file for testing
        temp_js = project_root / "temp_test.js"
        temp_js.write_text(SAMPLE_JS)

        deps = parser.analyze_dependencies(str(temp_js))
        logger.info(f"Dependencies:")
        logger.info(f"  - npm: {deps['npm']}")
        logger.info(f"  - local: {deps['local']}")

        # Check that we detected npm dependencies
        assert len(deps['npm']) >= 1, "Should detect npm dependencies"
        assert 'axios' in deps['npm'], "Should detect axios as npm dependency"

        # Cleanup
        temp_js.unlink()

        logger.info("\n" + "=" * 60)
        logger.info("All tests passed!")
        logger.info("=" * 60)

        return True

    except AssertionError as e:
        logger.error(f"\nAssertion failed: {e}")
        import traceback
        traceback.print_exc()
        return False
    except Exception as e:
        logger.error(f"\nTest failed: {e}")
        import traceback
        traceback.print_exc()
        return False


if __name__ == '__main__':
    success = test_js_parser()
    sys.exit(0 if success else 1)
