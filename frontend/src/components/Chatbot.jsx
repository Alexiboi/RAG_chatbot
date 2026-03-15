import { useEffect, useMemo, useState } from 'react';
import { useImmer } from 'use-immer';
import ChatMessages from '@/components/ChatMessages';
import ChatInput from '@/components/ChatInput';
import * as api from '@/api';

function Chatbot() {
  /**
   * chats: stores list of all chat sessions with {id: 'xxx', title: 'meeting notes'}
   * activeChatId: stores which chat is currently selected
   * messages: stores messages currently displayed in the main chat area
   * newMessage: message you currently have typed in the input box not yet submitted
   * isBootstrapping: True if app is still loading the messages from backend
   */
  const [chats, setChats] = useImmer([]);
  const [activeChatId, setActiveChatId] = useState(null);
  const [messages, setMessages] = useImmer([]);
  const [newMessage, setNewMessage] = useState('');
  const [isBootstrapping, setIsBootstrapping] = useState(true);
  const [isSettingsOpen, setIsSettingsOpen] = useState(false);
  const [isDarkMode, setIsDarkMode] = useState(false);

  // isLoading is True if the last message in the chat is currently loading.
  // used to stop user sending a message while the assistant is responding/loading
  const isLoading =
    messages.length > 0 && messages[messages.length - 1].loading;

  /**
   * activeChat: becomes the chat (e.g. {id: 'xxx', title: 'meeting'}) with id of activeChatId
   * useMemo avoids recomputation unnecessarily
   */
  const activeChat = useMemo(
    () => chats.find(chat => chat.id === activeChatId) || null,
    [chats, activeChatId]
  );

  useEffect(() => {
    async function loadInitialChats() {
      try {
        // loads all existing chats
        const existingChats = await api.getChats();
        setChats(existingChats);

        /** 
         * If chats already exist load the first one and set that to be active
         * and load the messages for that chatId
         * If no chats exist then create a new one 
        */
        if (existingChats.length > 0) {
          const firstChatId = existingChats[0].id;
          setActiveChatId(firstChatId);

          const chatMessages = await api.getChatMessages(firstChatId);
          setMessages(chatMessages);
        } else {
          const newChat = await api.createChat();
          setChats([newChat]);
          setActiveChatId(newChat.id);
          setMessages([]);
        }
      } catch (err) {
        console.error('Failed to load chats:', err);
      } finally {
        setIsBootstrapping(false); // user can now type in their query
      }
    }

    // calling function we just defined
    loadInitialChats();
  }, [setChats, setMessages]);

  /**
   * This method runs when the user clicks New Chat
   */
  async function handleCreateNewChat() {
    // if the assistant is responding then can't create a new chat
    if (isLoading) return;

    try {
      const newChat = await api.createChat();

      // unshift adds chat at the top of the list/start of the array
      // wouldn't make sends for the newest chat to be at the bottom
      setChats(draft => {
        draft.unshift(newChat);
      });

      setActiveChatId(newChat.id);
      setMessages([]);
    } catch (err) {
      console.error('Failed to create chat:', err);
    }
  }

  // This runs when the user clicks a chat in the sidebar
  async function handleSelectChat(chatId) {
    if (chatId === activeChatId || isLoading) return;

    try {
      setActiveChatId(chatId);
      const chatMessages = await api.getChatMessages(chatId);
      setMessages(chatMessages);
    } catch (err) {
      console.error('Failed to load chat messages:', err);
    }
  }

  async function submitNewMessage() {
    const trimmedMessage = newMessage.trim();
    // prevent submission if the input is empty, message is already being processed,
    // or there is no active chat
    if (!trimmedMessage || isLoading || !activeChatId) return;

    /**
     * Before calling backend, the UI immediatly adds:
     * the user's message and a placeholder assistant message that is loading
     */
    setMessages(draft => {
      draft.push(
        { role: 'user', content: trimmedMessage },
        { role: 'assistant', content: '', loading: true, error: false }
      );
    });

    // resets textarea after sending
    setNewMessage('');

    try {
      // calls chat_loop in backend with history from redis 
      // then stores user + assistant messages in Redis
      // then backend returns assistant answer here
      const data = await api.sendChatMessage(activeChatId, trimmedMessage);
      
      /**
       * Takes last message which was assistant placeholder message
       * and updates it with the real response
       */
      setMessages(draft => {
        const lastMessage = draft[draft.length - 1];
        lastMessage.content = data.answer; // use data.answer as data is a dictionary with answer, mode and retrieved
        lastMessage.loading = false;
        lastMessage.error = false;
      });

      // If the chat still has the default title New Chat, rename it using user's first message
      // this does not persist in Redis yet
      setChats(draft => {
        const chat = draft.find(c => c.id === activeChatId);
        if (chat && (!chat.title || chat.title === 'New Chat')) {
          chat.title = trimmedMessage.slice(0, 40);
        }
      });
    } catch (err) {
      console.error(err);

      setMessages(draft => {
        const lastMessage = draft[draft.length - 1];
        lastMessage.loading = false;
        lastMessage.error = true;
      });
    }
  }

  const toggleDarkMode = () => {
    setIsDarkMode(!isDarkMode);
    document.documentElement.classList.toggle('dark');
  };

  return (
    <div className="flex h-full min-h-0">
      <aside className="w-64 shrink-0 border-r border-primary-blue/20 bg-white/60 p-4 flex flex-col">
        <button
          onClick={handleCreateNewChat}
          className="mb-4 w-full rounded-xl bg-primary-blue px-4 py-3 text-left font-urbanist text-white transition hover:opacity-90"
        >
          + New chat
        </button>

        <div className="space-y-2 overflow-y-auto grow">
          {chats.map(chat => (
            <button
              key={chat.id}
              onClick={() => handleSelectChat(chat.id)}
              className={`block w-full rounded-xl px-3 py-3 text-left transition ${
                chat.id === activeChatId
                  ? 'bg-primary-blue/20 text-main-text'
                  : 'hover:bg-primary-blue/10 text-main-text/80'
              }`}
            >
              <div className="truncate font-medium">
                {chat.title || 'New Chat'}
              </div>
            </button>
          ))}
        </div>

        <button
          onClick={() => setIsSettingsOpen(true)}
          className="mt-4 w-full rounded-xl bg-primary-blue px-4 py-3 text-left font-urbanist text-white transition hover:opacity-90"
        >
          Settings
        </button>
      </aside>

      {isSettingsOpen && (
        <div className="fixed inset-0 bg-black/50 flex items-center justify-center z-50">
          <div className="bg-white rounded-lg p-6 w-96">
            <h2 className="text-xl font-semibold mb-4">Settings</h2>
            <div className="mb-4">
              <label className="flex items-center">
                <input
                  type="checkbox"
                  checked={isDarkMode}
                  onChange={toggleDarkMode}
                  className="mr-2"
                />
                <span>Dark Mode</span>
              </label>
            </div>
            <button
              onClick={() => setIsSettingsOpen(false)}
              className="bg-primary-blue text-white px-4 py-2 rounded hover:opacity-90"
            >
              Close
            </button>
          </div>
        </div>
      )}

      <div className="relative flex grow flex-col gap-6 pt-6 px-16 min-h-0">
        <header className='sticky top-0 shrink-0 z-20 bg-white pb-2'>
          <h1 className='font-urbanist text-[1.65rem] font-semibold'>{activeChat?.title || 'Tech Trends AI Chatbot'}</h1>
        </header>
        {isBootstrapping ? (
          <div className="mt-3 font-urbanist text-primary-blue text-xl font-light">
            Loading chats...
          </div>
        ) : messages.length === 0 ? (
          <div className="mt-3 font-urbanist text-primary-blue text-xl font-light space-y-2">
            <p>👋 Welcome!</p>
            <p>I have access to earnings calls from multiple companies and meeting notes</p>
            <p>I also have capability to interact with Jira and create issues using MCP</p>
            {!activeChat && <p>Create a new chat to get started.</p>}
          </div>
        ) : null}

        <ChatMessages
          messages={messages}
          isLoading={isLoading}
        />

        <ChatInput
          newMessage={newMessage}
          isLoading={isLoading || !activeChatId}
          setNewMessage={setNewMessage}
          submitNewMessage={submitNewMessage}
        />
      </div>
    </div>
  );
}

export default Chatbot;