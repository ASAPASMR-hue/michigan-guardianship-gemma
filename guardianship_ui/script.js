document.addEventListener('DOMContentLoaded', () => {
    const formPath = document.getElementById('form-path');
    const questionPath = document.getElementById('question-path');

    if (formPath) {
        formPath.addEventListener('click', () => {
            // Non-functional for now
        });
    }

    if (questionPath) {
        questionPath.addEventListener('click', () => {
            window.location.href = 'chatbot.html';
        });
    }
});