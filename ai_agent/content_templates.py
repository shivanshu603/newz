def get_template(template_name: str) -> 'BlogTemplate':
    """Get a template by name."""
    if template_name == 'default':
        return BlogTemplate()
    if template_name == 'standard_blog':
        return StandardBlogTemplate()  # Return a specific template for standard_blog




class BlogTemplate:
    def format_blog(self, topic: str, content: str) -> str:
        """Format the blog post with a title and content."""
        formatted_blog = f"# {topic}\n\n{content}"
        return formatted_blog

class StandardBlogTemplate(BlogTemplate):
    def format_blog(self, topic: str, content: str) -> str:
        """Format the blog post with a title, content, and additional elements."""
        formatted_blog = f"# {topic}\n\n{content}\n\n---\n\n*This is a standard blog post template.*"
        return formatted_blog
