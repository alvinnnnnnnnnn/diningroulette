document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const name = params.get('name');
    const location = params.get('location');
    
    document.getElementById('placeInfo').innerHTML = `
        <p><strong>Name:</strong> ${name}</p>
        <p><strong>Location:</strong> ${location}</p>
    `;

    document.getElementById('goBackButton').addEventListener('click', function() {
        window.location.href = `index.html`;
    });
});
