const places = [
    { type: "Restaurant", cuisines: "Chinese", location: "Tanjong Pagar", name: "Chengdu Bowl" },
    { type: "Restaurant", cuisines: "Korean", location: "Tanjong Pagar", name: "Noodle Star K" },
    { type: "Cafe", cuisines: "nil", location: "Tanjong Pagar", name: "Acoustics Coffee Bar" },
    { type: "Bar", cuisines: "Irish", location: "Raffles Place", name: "Molly Malones" }
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
            
            // Construct query parameters
            const queryParams = new URLSearchParams({
                name: place.name,
                type: place.type,
                cuisines: place.cuisines, // Join array into a string
                location: place.location
            }).toString();
            
            // Redirect to display.html with query parameters
            window.location.href = `display.html?${queryParams}`;
        } else {
            alert("No places found with the selected filters. Try different options!");
        }
    });
});
  
  
  