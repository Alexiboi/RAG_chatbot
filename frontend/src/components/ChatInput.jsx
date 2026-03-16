import useAutosize from '@/hooks/useAutosize';
import sendIcon from '@/assets/images/send.svg';

function ChatInput({ 
  newMessage,
  isLoading,
  setNewMessage,
  submitNewMessage,
  selectedMode,
  setSelectedMode 
}) {
  const textareaRef = useAutosize(newMessage);

  function handleKeyDown(e) {
    if(e.key === 'Enter' && !e.shiftKey && !isLoading) {
      e.preventDefault();
      submitNewMessage();
    }
  }
  
  return(
    <div className='sticky bottom-0 shrink-0 bg-white py-4'>
        <div className="mb-2 flex items-center justify-between px-2 text-sm">
          <span className="text-main-text/70">Mode</span>

          <select
            value={selectedMode}
            onChange={e => setSelectedMode(e.target.value)}
            className="rounded-md border border-primary-blue/20 bg-white px-2 py-1 text-main-text focus:outline-none"
            disabled={isLoading}
          >
            <option value="auto">Auto</option>
            <option value="llm">LLM</option>
            <option value="rag">RAG</option>
            <option value="mcp">MCP</option>
          </select>
        </div>

      <div className='p-1.5 bg-primary-blue/35 rounded-3xl z-50 font-mono origin-bottom animate-chat duration-400'>
        <div className='pr-0.5 bg-white relative shrink-0 rounded-3xl overflow-hidden ring-primary-blue ring-1 focus-within:ring-2 transition-all'>
          <textarea
            className='block w-full max-h-[140px] py-2 px-4 pr-11 bg-white rounded-3xl resize-none placeholder:text-primary-blue placeholder:leading-4 placeholder:-translate-y-1 sm:placeholder:leading-normal sm:placeholder:translate-y-0 focus:outline-hidden'
            ref={textareaRef}
            rows='1'
            value={newMessage}
            onChange={e => setNewMessage(e.target.value)}
            onKeyDown={handleKeyDown}
          />
          <button
            className='absolute top-1/2 -translate-y-1/2 right-3 p-1 rounded-md hover:bg-primary-blue/20'
            onClick={submitNewMessage}
          >
            <img src={sendIcon} alt='send' />
          </button>
        </div>
      </div>


    </div>
  );
}

export default ChatInput;