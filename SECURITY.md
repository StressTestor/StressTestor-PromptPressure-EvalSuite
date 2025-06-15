# Security Policy

## Supported Versions

| Version | Supported          |
| ------- | ------------------ |
| 1.5.x   | :white_check_mark: |
| < 1.5   | :x:                |

## Reporting a Vulnerability

We take security issues seriously and appreciate your efforts to responsibly disclose your findings.

### Reporting Process

1. **Do not** create a public GitHub issue for security vulnerabilities
2. Email your findings to [security@example.com](mailto:security@example.com)
3. Include a detailed description of the vulnerability
4. Include steps to reproduce the issue
5. If applicable, include proof of concept code

### Response Time

- We will acknowledge your email within 48 hours
- We will keep you informed of the progress towards fixing the vulnerability
- We will notify you when the vulnerability has been fixed

### Bug Bounty

While we don't currently have a formal bug bounty program, we are happy to acknowledge significant security improvements with a mention in our release notes (unless you prefer to remain anonymous).

## Security Updates

Security updates will be released as patch versions (e.g., 1.5.1, 1.5.2). We recommend always running the latest patch version of the library.

## Best Practices

### For Users
- Always keep your dependencies up to date
- Never commit API keys or sensitive information to version control
- Use environment variables for configuration
- Regularly rotate your API keys

### For Contributors
- Follow secure coding practices
- Validate all inputs
- Use parameterized queries to prevent SQL injection
- Keep dependencies up to date
- Run security scanners as part of your development workflow

## Dependencies

We regularly update our dependencies to include the latest security patches. You can check for known vulnerabilities in our dependencies using:

```bash
pip install safety
safety check
```

## Encryption

All data in transit is encrypted using TLS 1.2+.

## Reporting Security Issues in Dependencies

If you find a security issue in one of our dependencies, please report it to the maintainers of that package first. Once the issue is fixed, please let us know so we can update our dependencies.
