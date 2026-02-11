function login() {
  alert("Login successful (demo)");
}

function checkFraud() {
  let amount = document.getElementById("amount").value;

  fetch("/predict", {
    method: "POST",
    headers: {
      "Content-Type": "application/json"
    },
    body: JSON.stringify({ amount: amount })
  })
  .then(res => res.json())
  .then(data => {
    let output = document.getElementById("output");

    if (data.prediction === "Fraud") {
      output.innerHTML = "⚠️ Fraud Transaction Detected";
      output.style.color = "red";
    } else {
      output.innerHTML = "✅ Transaction is Safe";
      output.style.color = "lightgreen";
    }
  });
}