-- Table: public.chat_messages_history

-- DROP TABLE IF EXISTS public.chat_messages_history;

CREATE TABLE IF NOT EXISTS public.chat_messages_history
(
    id character(14) COLLATE pg_catalog."default" NOT NULL,
    session_id character(14) COLLATE pg_catalog."default" NOT NULL,
    sender_id character(14) COLLATE pg_catalog."default" NOT NULL,
    sender_type text COLLATE pg_catalog."default" NOT NULL,
    message text COLLATE pg_catalog."default" NOT NULL,
    created_at bigint,
    deleted_at bigint,
    updated_at bigint,
    CONSTRAINT chat_messages_history_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.chat_messages_history
    OWNER to postgres;

COMMENT ON TABLE public.chat_messages_history
    IS 'Stores chat messages between bots and users';

COMMENT ON COLUMN public.chat_messages_history.id
    IS 'Unique message identifier (primary key)';

COMMENT ON COLUMN public.chat_messages_history.session_id
    IS 'Chat session identifier';

COMMENT ON COLUMN public.chat_messages_history.sender_id
    IS 'ID of the message sender';

COMMENT ON COLUMN public.chat_messages_history.sender_type
    IS 'Type of sender (bot/user)';

COMMENT ON COLUMN public.chat_messages_history.message
    IS 'Message content';

COMMENT ON COLUMN public.chat_messages_history.created_at
    IS 'Timestamp when message was created';

COMMENT ON COLUMN public.chat_messages_history.deleted_at
    IS 'Timestamp when message was deleted (soft delete)';

COMMENT ON COLUMN public.chat_messages_history.updated_at
    IS 'Timestamp when message was last updated';
-- Index: idx_chat_messages_history_session_id

-- DROP INDEX IF EXISTS public.idx_chat_messages_history_session_id;

CREATE INDEX IF NOT EXISTS idx_chat_messages_history_session_id
    ON public.chat_messages_history USING btree
    (session_id COLLATE pg_catalog."default" ASC NULLS LAST)
    WITH (fillfactor=100, deduplicate_items=True)
    TABLESPACE pg_default;
