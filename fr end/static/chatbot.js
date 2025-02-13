const chatBox = document.getElementById("chat-box");
const userInput = document.getElementById("user-input");
const sendBtn = document.getElementById("send-btn");

sendBtn.addEventListener("click", () => {
    const message = userInput.value.trim();
    if (message) {
        addMessage("You", message);
        userInput.value = "";
        fetch("/chat", {
            method: "POST",
            headers: {
                "Content-Type": "application/json",
            },
            body: JSON.stringify({ message }),
        })
            .then((response) => response.json())
            .then((data) => addMessage("Bot", data.response))
            .catch(() =>
                addMessage("Bot", "Sorry, something went wrong. Please try again.")
            );
    }
});

function addMessage(sender, message) {
    const messageDiv = document.createElement("div");
    messageDiv.textContent = `${sender}: ${message}`;
    chatBox.appendChild(messageDiv);
    chatBox.scrollTop = chatBox.scrollHeight;
}
