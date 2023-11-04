
const runButton = document.getElementById('run-button');
const urlInput = document.getElementById('url-input');
const summaryResultDiv = document.getElementById('summary-result');
const summaryTextDiv = document.getElementById('summary-text');
const privacyResultDiv = document.getElementById('privacy-result');
const privacyTextDiv = document.getElementById('privacy-text');
// In your Chrome extension's content script or popup script

document.addEventListener('DOMContentLoaded', function() {
const inp = document.getElementById('url-input');
console.log('here')
chrome.tabs.query({ active: true, currentWindow: true }, (tabs) => {
if (tabs.length > 0) {
  const currentTab = tabs[0];
  const currentUrl = currentTab.url;
console.log(currentUrl)

  inp.value = currentUrl;
}
});
});


runButton.addEventListener('click', handleRunButtonClick);

function handleRunButtonClick() {
  const url = urlInput.value;  // Get URL from input field
  console.log(url)
  const requestBodyForSummary = { url: url };
  const requestBodyForPrivacy = { url: url };

  // Using 'no-cors' mode for testing purposes
  fetch('https://n30cfr1w-8080.inc1.devtunnels.ms/summarize', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(requestBodyForSummary)
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(response);
    }
    return response.json();
  })
  .then(data => {
    summaryResultDiv.innerText = summaryResultDiv.innerText  + data.text;
  })
  .catch(error => {
    console.log("Summary error:", error);
  });
  // Using 'no-cors' mode for testing purposes
  fetch('https://n30cfr1w-7000.inc1.devtunnels.ms/rating', {
    method: 'POST',
    headers: {
      'Content-Type': 'application/json'
    },
    body: JSON.stringify(requestBodyForPrivacy)
  })
  .then(response => {
    if (!response.ok) {
      throw new Error(response);
    }
    return response.json();
  })
  .then(data => {
    let perc = data *100
    console.log(perc)

    if (perc < 50) {
      privacyResultDiv.style.background = 'linear-gradient(45deg, #8B0000, #FF4500)';


    } else if (perc < 65) {

      privacyResultDiv.style.background = 'linear-gradient(45deg, #FF5733, #FF0000)';



    }else if(perc>80){
      privacyResultDiv.style.background = 'linear-gradient(45deg, #FFD700, #FFA500)';

    }
    
    
    privacyResultDiv.innerText='Privacy Policy Result:'
    privacyResultDiv.innerText=privacyResultDiv.innerText + ' ' + ( data * 100).toFixed(2) + '%';
  })
  .catch(error => {
    
    console.log("Privacy error:", error);
  });
}