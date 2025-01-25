import React, { useState, useEffect, useRef } from "react";
import styled from "styled-components";
import pngImage from "../../assets/images/pageUfsc.png";
import pngImage1 from "../../assets/images/botUfsc.png";

export function ChatUfscPage() {
  const [isHovering, setIsHovering] = useState(false);
  const [isChatOpen, setIsChatOpen] = useState(false);
  const [message, setMessage] = useState("");
  const [chatMessages, setChatMessages] = useState([]);
  const [isLoading, setIsLoading] = useState(false);
  const messagesContainerRef = useRef(null);

  const toggleChat = () => {
    setIsChatOpen(!isChatOpen);
  };

  const sendMessage = async () => {
    if (message.trim()) {
      // Add user message to chat
      setChatMessages([...chatMessages, { text: message, isUser: true }]);
      setMessage("");
      setIsLoading(true);
  
      try {
        const response = await fetch("http://localhost:8000/search/query", {
          method: "POST",
          headers: {
            "Content-Type": "application/json",
          },
          body: JSON.stringify({
            text: message.trim(),
            top_n: 5,
          }),
        });
  
        // Handle the streamed response
        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let accumulatedResponse = "";
  
        while (true) {
          const { done, value } = await reader.read();
          if (done) break;
  
          // Decode the chunk and append it to the accumulated response
          const chunk = decoder.decode(value, { stream: true });
          accumulatedResponse += chunk;
  
          // Update the chat UI with the accumulated response
          setChatMessages((prevMessages) => {
            const lastMessage = prevMessages[prevMessages.length - 1];
            if (lastMessage && !lastMessage.isUser) {
              // Update the last assistant message
              return [
                ...prevMessages.slice(0, -1),
                { text: accumulatedResponse, isUser: false },
              ];
            } else {
              // Add a new assistant message
              return [...prevMessages, { text: accumulatedResponse, isUser: false }];
            }
          });
        }
      } catch (error) {
        console.error('Error:', error);
        alert('Error fetching data. Please try again.');
      } finally {
        setIsLoading(false);
      }
    }
  };

  const handleKeyPress = (e) => {
    if (e.key === "Enter") {
      sendMessage();
    }
  };

  useEffect(() => {
    if (messagesContainerRef.current) {
      messagesContainerRef.current.scrollTop = messagesContainerRef.current.scrollHeight;
    }
  }, [chatMessages, isChatOpen]);

  return (
    <Container>
      <Image src={pngImage} alt="notices" />
      <ContentDiv
        onMouseEnter={() => setIsHovering(true)}
        onMouseLeave={() => setIsHovering(false)}
        onClick={toggleChat}
        hovering={isHovering}
      >
        {isChatOpen && (
          <ChatBox onClick={(e) => e.stopPropagation()}>
            <ChatBoxHeader>
              <ChatBoxTitle>ChatUFSC</ChatBoxTitle>
              <div>
                <ClearButton onClick={() => setChatMessages([])}>Clear Chat</ClearButton>
                <CloseButton onClick={toggleChat}>Fechar</CloseButton>
              </div>
            </ChatBoxHeader>
            <MessagesContainer ref={messagesContainerRef}>
              {chatMessages.map((msg, index) => (
                <Message key={index} text={msg.text} isUser={msg.isUser} />
              ))}
              {isLoading && <LoadingText>Pensando...</LoadingText>}
            </MessagesContainer>
            <ChatInputContainer>
              <ChatInput
                type="text"
                placeholder="Digite sua mensagem..."
                value={message}
                onChange={(e) => setMessage(e.target.value)}
                onKeyPress={handleKeyPress}
              />
              <SendButton onClick={sendMessage}>Enviar</SendButton>
            </ChatInputContainer>
          </ChatBox>
        )}
      </ContentDiv>
    </Container>
  );
}

const Message = ({ text, isUser }) => {
  const handleCopy = () => {
    navigator.clipboard.writeText(text).then(() => {
      alert("Message copied to clipboard!");
    });
  };

  return (
    <MessageContainer isUser={isUser}>
      <MessageText>{text}</MessageText>
      <CopyButton onClick={handleCopy}>Copy</CopyButton>
    </MessageContainer>
  );
};

const Container = styled.div`
  position: relative;
  width: 100%;
  height: 100%;
`;

const Image = styled.img`
  width: 100%;
  height: 100%;
  object-fit: cover;
`;

const ContentDiv = styled.div`
  position: absolute;
  top: 90%;
  left: 90%;
  transform: translate(-50%, -50%);
  width: 60px;
  height: 60px;
  border-radius: 180px;
  background: url(${pngImage1});
  background-size: cover;
  border: 2px solid black;
  filter: ${({ hovering }) => (hovering ? "none" : "brightness(70%)")};
  transition: filter 0.3s ease-in-out;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
`;

const ChatBox = styled.div`
  position: absolute;
  bottom: 120%;
  right: 0;
  background-color: #f9f9f9;
  border: 1px solid #ccc;
  border-radius: 10px;
  padding: 10px;
  width: 400px;
  height: 500px;
  display: flex;
  flex-direction: column;
`;

const ChatBoxHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 10px;
`;

const ChatBoxTitle = styled.h3`
  font-size: 18px;
  margin: 0;
`;

const MessagesContainer = styled.div`
  flex: 1;
  padding: 10px;
  border: 1px solid #ccc;
  border-radius: 10px;
  background-color: #f9f9f9;
  box-shadow: 0 2px 10px rgba(0, 0, 0, 0.1);
  overflow-y: auto;
  display: flex;
  flex-direction: column;
`;

const MessageContainer = styled.div`
  background-color: ${props => props.isUser ? '#007bff' : '#f1f0f0'};
  color: ${props => props.isUser ? 'white' : 'black'};
  padding: 10px 15px;
  margin-bottom: 10px;
  border-radius: 20px;
  max-width: 80%;
  align-self: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  word-wrap: break-word;
  box-shadow: 0 2px 5px rgba(0, 0, 0, 0.1);
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const MessageText = styled.div`
  flex: 1;
`;

const CopyButton = styled.button`
  background: none;
  border: none;
  color: ${props => props.isUser ? 'white' : '#007bff'};
  cursor: pointer;
  margin-left: 10px;
  font-size: 12px;
  &:hover {
    text-decoration: underline;
  }
`;

const ChatInputContainer = styled.div`
  display: flex;
  margin-top: 10px;
`;

const ChatInput = styled.input`
  flex: 1;
  height: 40px;
  border: 1px solid #ccc;
  border-radius: 5px;
  padding: 5px;
  margin-right: 5px;
`;

const SendButton = styled.button`
  width: 100px;
  height: 40px;
  background-color: #1659bf;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
`;

const ClearButton = styled.button`
  height: 40px;
  background-color: #ff4d4d;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  margin-right: 10px;
  padding: 0 20px;
  &:hover {
    background-color: #ff1a1a;
  }
`;

const CloseButton = styled.button`
  height: 40px;
  background-color: #1659bf;
  color: white;
  border: none;
  border-radius: 5px;
  cursor: pointer;
  margin-bottom: 10px;
  padding: 0 20px;
`;

const LoadingText = styled.p`
  font-style: italic;
  color: #888;
`;