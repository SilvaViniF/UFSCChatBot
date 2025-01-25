import React, { useState, useEffect, useRef } from "react";
import styled, { keyframes } from "styled-components";
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

        const reader = response.body.getReader();
        const decoder = new TextDecoder("utf-8");
        let accumulatedResponse = "";

        while (true) {
          const { done, value } = await reader.read();
          if (done) break;

          const chunk = decoder.decode(value, { stream: true });
          accumulatedResponse += chunk;

          setChatMessages((prevMessages) => {
            const lastMessage = prevMessages[prevMessages.length - 1];
            if (lastMessage && !lastMessage.isUser) {
              return [
                ...prevMessages.slice(0, -1),
                { text: accumulatedResponse, isUser: false },
              ];
            } else {
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

  const handleInputChange = (e) => {
    if (e.target.value.length <= 512) {
      setMessage(e.target.value);
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
              <CloseButton onClick={toggleChat}>Fechar</CloseButton>
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
                onChange={handleInputChange}
                onKeyPress={handleKeyPress}
                maxLength={512} // Set max length to 512 characters
              />
              <CharacterCount>
                {message.length}/512
              </CharacterCount>
              <ButtonGroup>
                <ClearButton onClick={() => setChatMessages([])}>Limpar</ClearButton>
                <SendButton onClick={sendMessage}>Enviar</SendButton>
              </ButtonGroup>
            </ChatInputContainer>
          </ChatBox>
        )}
      </ContentDiv>
    </Container>
  );
}

const Message = ({ text, isUser }) => {
  const handleCopy = () => {
    navigator.clipboard.writeText(text);
  };

  return (
    <MessageContainer isUser={isUser}>
      <MessageText isUser={isUser}>{text}</MessageText>
      <CopyButton isUser={isUser} onClick={handleCopy}>Copiar</CopyButton>
    </MessageContainer>
  );
};

// Animations
const fadeIn = keyframes`
  from { opacity: 0; transform: translateY(10px); }
  to { opacity: 1; transform: translateY(0); }
`;

// Styles
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
  border-radius: 50%;
  background: url(${pngImage1});
  background-size: cover;
  border: 2px solid #1659bf;
  filter: ${({ hovering }) => (hovering ? "brightness(100%)" : "brightness(70%)")};
  transition: filter 0.3s ease-in-out, transform 0.3s ease-in-out;
  display: flex;
  justify-content: center;
  align-items: center;
  cursor: pointer;
  &:hover {
    transform: translate(-50%, -50%) scale(1.1);
  }
`;

const ChatBox = styled.div`
  position: absolute;
  bottom: 120%;
  right: 0;
  background: #ffffff;
  border-radius: 20px;
  padding: 20px;
  width: 500px; /* Larger chat box */
  height: 600px; /* Larger chat box */
  display: flex;
  flex-direction: column;
  box-shadow: 0 10px 30px rgba(0, 0, 0, 0.1);
  animation: ${fadeIn} 0.3s ease-in-out;
  border: 1px solid #e0e0e0;
`;

const ChatBoxHeader = styled.div`
  display: flex;
  justify-content: space-between;
  align-items: center;
  padding-bottom: 15px;
  border-bottom: 1px solid #e0e0e0;
`;

const ChatBoxTitle = styled.h3`
  font-size: 24px; /* Larger font size */
  margin: 0;
  color: #1659bf;
  font-family: 'Poppins', sans-serif;
  font-weight: 600; /* Bold font */
`;

const MessagesContainer = styled.div`
  flex: 1;
  padding: 10px;
  overflow-y: auto;
  display: flex;
  flex-direction: column;
  gap: 10px;
`;

const MessageContainer = styled.div`
  background: ${props => props.isUser ? '#1659bf' : '#f1f1f1'};
  color: ${props => props.isUser ? '#ffffff' : '#333333'};
  padding: 12px 16px;
  border-radius: 15px;
  max-width: 80%;
  align-self: ${props => props.isUser ? 'flex-end' : 'flex-start'};
  word-wrap: break-word;
  box-shadow: 0 4px 10px rgba(0, 0, 0, 0.1);
  animation: ${fadeIn} 0.3s ease-in-out;
  display: flex;
  justify-content: space-between;
  align-items: center;
`;

const MessageText = styled.div`
  flex: 1;
  font-family: 'Inter', sans-serif;
`;

const CopyButton = styled.button`
  background: none;
  border: none;
  color: ${props => props.isUser ? '#ffffff' : '#1659bf'};
  cursor: pointer;
  margin-left: 10px;
  font-size: 12px;
  &:hover {
    opacity: 0.8;
  }
`;

const ChatInputContainer = styled.div`
  display: flex;
  margin-top: 15px;
  gap: 10px;
  align-items: center;
`;

const ChatInput = styled.input`
  flex: 1;
  height: 40px;
  border: 1px solid #e0e0e0;
  border-radius: 10px;
  padding: 10px;
  background: #ffffff;
  color: #333333;
  font-family: 'Inter', sans-serif;
  &:focus {
    outline: none;
    border-color: #1659bf;
    box-shadow: 0 0 0 2px rgba(22, 89, 191, 0.2);
  }
`;

const CharacterCount = styled.span`
  font-size: 12px;
  color: #888;
  font-family: 'Inter', sans-serif;
`;

const ButtonGroup = styled.div`
  display: flex;
  gap: 10px;
`;

const SendButton = styled.button`
  width: 100px;
  height: 40px;
  background: #1659bf;
  color: #ffffff;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  &:hover {
    background: #134a9e;
  }
`;

const ClearButton = styled.button`
  width: 100px;
  height: 40px;
  background: #ff4d4d;
  color: white;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  &:hover {
    background: #e04444;
  }
`;

const CloseButton = styled.button`
  height: 40px;
  background: #1659bf;
  color: white;
  border: none;
  border-radius: 10px;
  cursor: pointer;
  padding: 0 20px;
  font-family: 'Poppins', sans-serif;
  font-weight: 600;
  &:hover {
    background: #134a9e;
  }
`;

const LoadingText = styled.p`
  font-style: italic;
  color: #888;
  font-family: 'Inter', sans-serif;
`;