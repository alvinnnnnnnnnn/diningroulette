const places = [
    { type: "Restaurant", cuisines: "Chinese", location: "Tanjong Pagar", name: "Chengdu Bowl"},
    { type: "Restaurant", cuisines: "Korean", location: "Tanjong Pagar", name: "Noodle Star K"},
    { type: "Cafe", cuisines: "Japanese", location: "Tanjong Pagar", name: "Acoustics Coffee Bar"}
    // Add more places as needed, with multiple cuisines where applicable
];

document.addEventListener('DOMContentLoaded', (event) => { 
    document.getElementById('findButton').addEventListener('click', function() {
        const type = document.getElementById('type').value;
        const cuisine = document.getElementById('cuisines').value;
        const location = document.getElementById('location').value;
    
        const filteredPlaces = places.filter(place => {
            return (!type || place.type === type) && 
                   (!cuisine || place.cuisines == cuisine) && 
                   (!location || place.location === location)
        });
    
        if (filteredPlaces.length > 0) {
            const randomIndex = Math.floor(Math.random() * filteredPlaces.length);
            const place = filteredPlaces[randomIndex];
            document.getElementById('result').textContent = `How about ${place.name}, which is a ${place.type} offering ${place.cuisines} cuisine in ${place.location}?`;
        } else {
            document.getElementById('result').textContent = "No places found with the selected filters. Try different options!";
        }
    });
});
  
  
  