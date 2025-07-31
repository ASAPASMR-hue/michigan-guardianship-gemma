// Michigan Guardianship AI - Chat Interface JavaScript

// Global variables
let isProcessing = false;

// DOM elements
const chatForm = document.getElementById('chatForm');
const questionInput = document.getElementById('questionInput');
const sendButton = document.getElementById('sendButton');
const messagesContainer = document.getElementById('messagesContainer');
const welcomeMessage = document.getElementById('welcomeMessage');
const loadingIndicator = document.getElementById('loadingIndicator');
const errorModal = document.getElementById('errorModal');
const errorMessage = document.getElementById('errorMessage');

// Initialize on page load
document.addEventListener('DOMContentLoaded', function() {
    // Set up form submission
    chatForm.addEventListener('submit', handleSubmit);
    
    // Set up example questions
    const exampleQuestions = document.querySelectorAll('.input-hints li');
    exampleQuestions.forEach(example => {
        example.addEventListener('click', function() {
            questionInput.value = this.textContent.replace(/['"]/g, '');
            questionInput.focus();
        });
    });
    
    // Auto-resize textarea
    questionInput.addEventListener('input', function() {
        this.style.height = 'auto';
        this.style.height = Math.min(this.scrollHeight, 120) + 'px';
    });
    
    // Focus on input
    questionInput.focus();
});

// Handle form submission
async function handleSubmit(e) {
    e.preventDefault();
    
    if (isProcessing) return;
    
    const question = questionInput.value.trim();
    if (!question) return;
    
    // Hide welcome message on first question
    if (welcomeMessage.style.display !== 'none') {
        welcomeMessage.style.display = 'none';
    }
    
    // Add user message to chat
    addMessage(question, 'user');
    
    // Clear input and disable form
    questionInput.value = '';
    questionInput.style.height = 'auto';
    setProcessingState(true);
    
    try {
        // Show loading indicator
        loadingIndicator.style.display = 'flex';
        
        // Send request to API
        const response = await fetch('/api/ask', {
            method: 'POST',
            headers: {
                'Content-Type': 'application/json',
            },
            body: JSON.stringify({ question: question })
        });
        
        const data = await response.json();
        
        if (!response.ok) {
            throw new Error(data.details || data.error || 'Failed to get response');
        }
        
        // Add assistant response to chat
        addMessage(data.answer, 'assistant', data.metadata);
        
    } catch (error) {
        console.error('Error:', error);
        showError(error.message || 'Failed to connect to the server. Please try again.');
    } finally {
        // Hide loading indicator and re-enable form
        loadingIndicator.style.display = 'none';
        setProcessingState(false);
        questionInput.focus();
    }
}

// Add message to chat
function addMessage(content, sender, metadata = null) {
    const messageDiv = document.createElement('div');
    messageDiv.className = `message ${sender}`;
    
    // Create icon
    const iconDiv = document.createElement('div');
    iconDiv.className = 'message-icon';
    iconDiv.innerHTML = sender === 'user' 
        ? '<i class="fas fa-user"></i>' 
        : '<i class="fas fa-robot"></i>';
    
    // Create content container
    const contentDiv = document.createElement('div');
    contentDiv.className = 'message-content';
    
    // Process content for markdown-like formatting
    const processedContent = processContent(content);
    contentDiv.innerHTML = processedContent;
    
    // Add timestamp
    const timeDiv = document.createElement('div');
    timeDiv.className = 'message-time';
    timeDiv.textContent = new Date().toLocaleTimeString();
    contentDiv.appendChild(timeDiv);
    
    // Add metadata if available (for debugging)
    if (metadata && window.location.search.includes('debug')) {
        const metaDiv = document.createElement('details');
        metaDiv.innerHTML = `
            <summary style="cursor: pointer; font-size: 0.8em; margin-top: 8px;">Debug Info</summary>
            <pre style="font-size: 0.7em; margin-top: 4px;">${JSON.stringify(metadata, null, 2)}</pre>
        `;
        contentDiv.appendChild(metaDiv);
    }
    
    // Assemble message
    messageDiv.appendChild(iconDiv);
    messageDiv.appendChild(contentDiv);
    
    // Add to container and scroll to bottom
    messagesContainer.appendChild(messageDiv);
    messagesContainer.scrollTop = messagesContainer.scrollHeight;
}

// Process content for basic markdown formatting
function processContent(content) {
    // Escape HTML
    content = escapeHtml(content);
    
    // Convert markdown-like formatting
    // Bold
    content = content.replace(/\*\*(.*?)\*\*/g, '<strong>$1</strong>');
    
    // Italic
    content = content.replace(/\*(.*?)\*/g, '<em>$1</em>');
    
    // Links
    content = content.replace(/\[([^\]]+)\]\(([^)]+)\)/g, '<a href="$2" target="_blank" rel="noopener">$1</a>');
    
    // Line breaks
    content = content.replace(/\n\n/g, '</p><p>');
    content = content.replace(/\n/g, '<br>');
    
    // Wrap in paragraphs
    content = '<p>' + content + '</p>';
    
    // Lists
    content = content.replace(/<p>(\s*[-*]\s+.+)(<br>[-*]\s+.+)*<\/p>/g, function(match) {
        const items = match
            .replace(/<\/?p>/g, '')
            .split(/<br>/)
            .filter(item => item.trim())
            .map(item => '<li>' + item.replace(/^[-*]\s+/, '') + '</li>')
            .join('');
        return '<ul>' + items + '</ul>';
    });
    
    // Numbered lists
    content = content.replace(/<p>(\s*\d+\.\s+.+)(<br>\d+\.\s+.+)*<\/p>/g, function(match) {
        const items = match
            .replace(/<\/?p>/g, '')
            .split(/<br>/)
            .filter(item => item.trim())
            .map(item => '<li>' + item.replace(/^\d+\.\s+/, '') + '</li>')
            .join('');
        return '<ol>' + items + '</ol>';
    });
    
    // Code blocks
    content = content.replace(/```([\s\S]*?)```/g, '<pre><code>$1</code></pre>');
    
    // Inline code
    content = content.replace(/`([^`]+)`/g, '<code>$1</code>');
    
    return content;
}

// Escape HTML to prevent XSS
function escapeHtml(text) {
    const map = {
        '&': '&amp;',
        '<': '&lt;',
        '>': '&gt;',
        '"': '&quot;',
        "'": '&#39;'
    };
    return text.replace(/[&<>"']/g, m => map[m]);
}

// Set processing state
function setProcessingState(processing) {
    isProcessing = processing;
    questionInput.disabled = processing;
    sendButton.disabled = processing;
    
    if (processing) {
        sendButton.innerHTML = '<i class="fas fa-spinner fa-spin"></i> <span>Sending...</span>';
    } else {
        sendButton.innerHTML = '<i class="fas fa-paper-plane"></i> <span>Send</span>';
    }
}

// Show error modal
function showError(message) {
    errorMessage.textContent = message;
    errorModal.style.display = 'flex';
}

// Close error modal
function closeErrorModal() {
    errorModal.style.display = 'none';
}

// Close modal when clicking outside
window.onclick = function(event) {
    if (event.target === errorModal) {
        closeErrorModal();
    }
}

// Handle keyboard shortcuts
document.addEventListener('keydown', function(e) {
    // Ctrl/Cmd + Enter to submit
    if ((e.ctrlKey || e.metaKey) && e.key === 'Enter' && !isProcessing) {
        e.preventDefault();
        chatForm.dispatchEvent(new Event('submit'));
    }
});