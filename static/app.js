// Voice Agents Project - Day 1 (Basic Version)
// Simple JavaScript for basic functionality

console.log('Voice Agents Project Day 1 - Loaded Successfully!');

// Simple function to test JavaScript
function showMessage() {
    alert('JavaScript is working! 🎉');
    console.log('JavaScript test button clicked');
}

// Simple function to test API
async function testAPI() {
    try {
        const response = await fetch('/api/hello');
        const data = await response.json();
        
        const responseDiv = document.getElementById('response');
        responseDiv.style.display = 'block';
        responseDiv.innerHTML = `
            <strong>API Response:</strong><br>
            ${JSON.stringify(data, null, 2)}
        `;
        
        console.log('API Response:', data);
    } catch (error) {
        const responseDiv = document.getElementById('response');
        responseDiv.style.display = 'block';
        responseDiv.innerHTML = `<strong>Error:</strong> ${error.message}`;
        console.error('API Error:', error);
    }
}

// Log success message
console.log('✅ Day 1 Complete: FastAPI + HTML + JavaScript working together!');
