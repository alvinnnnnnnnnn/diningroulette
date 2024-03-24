const places = [
    { type: "Restaurant", cuisines: ["Chinese"], location: "Tanjong Pagar", name: "Chengdu Bowl", halalCertified: false },
    { type: "Restaurant", cuisines: ["Korean"], location: "Tanjong Pagar", name: "Noodle Star K", halalCertified: false },
    { type: "Cafe", cuisines: ["Japanese"], location: "Tanjong Pagar", name: "Acoustics Coffee Bar", halalCertified: false  }
    // Add more places as needed, with multiple cuisines where applicable
];
  
document.getElementById('findButton').addEventListener('click', function() {
    const type = document.getElementById('type').value;
    const cuisine = document.getElementById('cuisine').value;
    const location = document.getElementById('location').value;
    const halalCertified = document.getElementById('halal').checked;
  
    const filteredPlaces = places.filter(place => {
        return (!type || place.type === type) && 
               (!cuisine || place.cuisines.includes(cuisine)) && 
               (!location || place.location === location) && 
               (!halalCertified || place.halalCertified === halalCertified);
});
  
    if (filteredPlaces.length > 0) {
        const randomIndex = Math.floor(Math.random() * filteredPlaces.length);
        const place = filteredPlaces[randomIndex];
        document.getElementById('result').textContent = `How about ${place.name}, which is a ${place.type} offering ${place.cuisines.join(' and ')} cuisine in ${place.location}? ${place.halalCertified ? '(Halal Certified)' : ''}`;
    } else {
        document.getElementById('result').textContent = "No places found with the selected filters. Try different options!";
    }
});
  
  
  