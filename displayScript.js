document.addEventListener('DOMContentLoaded', () => {
    const params = new URLSearchParams(window.location.search);
    const name = params.get('name');
    const type = params.get('type');
    const cuisines = params.get('cuisines');
    const location = params.get('location');
    const halalCertified = params.get('halalCertified') === 'true' ? 'Yes' : 'No';
    
    document.getElementById('placeInfo').innerHTML = `
        <p><strong>Name:</strong> ${name}</p>
        <p><strong>Type:</strong> ${type}</p>
        <p><strong>Cuisines:</strong> ${cuisines}</p>
        <p><strong>Location:</strong> ${location}</p>
        <p><strong>Halal Certified:</strong> ${halalCertified}</p>
    `;
});