import { useState } from 'react';
import { useImmer } from 'use-immer';
import ChatMessages from '@/components/ChatMessages';
import ChatInput from '@/components/ChatInput';
import * as api from "@/api";

function Chatbot() {
    /**
     * chatId: stores the current chat session id
     * messages: holds all messages in current chat, each message contains a role (user/assistant), content, loading and error property
     * newMessage: Stores the current text in the chat input (before it gets submitted)
     */
    

  // useImmer updates state indirectly by creating a new object copy, as state updates must be performed immutably in react
  const [messages, setMessages] = useImmer([]);
  const [newMessage, setNewMessage] = useState('');


  const isLoading = messages.length && messages[messages.length - 1].loading;

  async function submitNewMessage() {
    const trimmedMessage = newMessage.trim();
    // make sure input message is not empty or that a response is loading before proceeding
    if (!trimmedMessage || isLoading) return;

    // add users message to the chat and placeholder assistant message
    setMessages(draft => [...draft,
        { role: 'user', content: trimmedMessage },
        { role: 'assistant', content: '', sources: [], loading: true, error: false }
    ]);
    setNewMessage('');

    try {
      const data = await api.sendMessage(trimmedMessage);

      setMessages(draft => {
        const lastMessage = draft[draft.length - 1];
        lastMessage.content = data.answer; // maybe data.response.answer depends on api response format
        lastMessage.loading = false;
        lastMessage.error = false;
      })
    } catch (err) {
        // if there are any errors we set the assistant message's error property True to display an error message in the chat
        console.log(err);
        setMessages(draft => {
        draft[draft.length - 1].loading = false;
        draft[draft.length - 1].error = true;
        });
    }
  }

  return (
    // If there are no messages in chat, display the welcome message
    <div className='relative grow flex flex-col gap-6 pt-6'>
    
      {messages.length === 0 && (
        <div className='mt-3 font-urbanist text-primary-blue text-xl font-light space-y-2'>
          <p>👋 Welcome!</p>
          <p>I have access to earnings calls from multiple companies and meeting notes</p>
          <p>I also have capability to interact with Jira and create Issues using MCP</p>
        </div>
      )}
      <ChatMessages
        messages={messages}
        isLoading={isLoading}
      />
      <ChatInput
        newMessage={newMessage}
        isLoading={isLoading}
        setNewMessage={setNewMessage}
        submitNewMessage={submitNewMessage}
      />
    </div>
  );
}

export default Chatbot;