document.getElementById("btn-predict").addEventListener("click", async function () {

    const btn = document.getElementById("btn-predict");
    const btnTxt = document.getElementById("btnTxt");

    // 🔹 Show loading state
    btnTxt.innerText = "Calculating...";
    btn.disabled = true;

    // 1 read values
    const bedrooms = document.getElementById("bedrooms").value;
    const bathrooms = document.getElementById("bathrooms").value;
    const stories = document.getElementById("stories").value;
    const parking = document.getElementById("parking").value;
    const area = document.getElementById("sqft").value;
    const mainroad = document.getElementById("mainroad").value;
    const guestroom = document.getElementById("guest_room").value;
    const basement = document.getElementById("basement").value;
    const hotwaterheating = document.getElementById("hot_water_heater").value;
    const airconditioning = document.getElementById("air_conditioning").value;
    const prefarea = document.getElementById("prefarea").value;
    const furnishingstatus = document.getElementById("furnishing").value;


    // 2 build payload
    const payload = {
        bedrooms: bedrooms,
        bathrooms: bathrooms,
        stories: stories,
        parking: parking,
        area: area,
        mainroad: mainroad,
        guestroom: guestroom,
        basement: basement,
        hotwaterheating: hotwaterheating,
        airconditioning: airconditioning,
        prefarea: prefarea,
        furnishingstatus: furnishingstatus
    }


    // 3 call API
    try {
        const response = await fetch("/predict", {
            method: "POST",
            headers: {
                "Content-Type": "application/json"
            },
            body: JSON.stringify(payload)
        });


        // 4 read response
        const data = await response.json();

        // 5 update UI
        document.getElementById("resultPrice").innerText =
            "Estimated Price: \n$" + Number(data.Prediction).toLocaleString();

        document.getElementById("resultCard").classList.add("visible");
    }

    
    catch (error) {
        console.error(error);
        alert("Prediction failed. Please try again.");
    }

    // 🔹 Restore button
    btnTxt.innerText = "Estimate Price →";
    btn.disabled = false;
});




// console.log(bedrooms);
// console.log(bathrooms);
// console.log(stories);
// console.log(parking);
// console.log(area);
// console.log(mainroad);
// console.log(guestroom);
// console.log(basement);
// console.log(hotwaterheating);
// console.log(airconditioning);
// console.log(prefarea);
// console.log(furnishingstatus);