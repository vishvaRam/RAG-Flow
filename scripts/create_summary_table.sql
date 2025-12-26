-- Table: public.chat_session_summaries

-- DROP TABLE IF EXISTS public.chat_session_summaries;

CREATE TABLE IF NOT EXISTS public.chat_session_summaries
(
    id character varying(14) COLLATE pg_catalog."default" NOT NULL,
    session_id character(14) COLLATE pg_catalog."default" NOT NULL,
    summary text COLLATE pg_catalog."default" NOT NULL,
    messages_count integer NOT NULL,
    created_at bigint NOT NULL,
    updated_at bigint,
    CONSTRAINT chat_session_summaries_pkey PRIMARY KEY (id)
)

TABLESPACE pg_default;

ALTER TABLE IF EXISTS public.chat_session_summaries
    OWNER to postgres;

COMMENT ON TABLE public.chat_session_summaries
    IS 'Stores summaries of chat sessions';

COMMENT ON COLUMN public.chat_session_summaries.id
    IS 'Unique identifier for the summary record';

COMMENT ON COLUMN public.chat_session_summaries.session_id
    IS 'Identifier for the chat session';

COMMENT ON COLUMN public.chat_session_summaries.summary
    IS 'Summary of the chat session';

COMMENT ON COLUMN public.chat_session_summaries.messages_count
    IS 'Number of messages in the chat session';

COMMENT ON COLUMN public.chat_session_summaries.created_at
    IS 'Timestamp when the summary was created';

COMMENT ON COLUMN public.chat_session_summaries.updated_at
    IS 'Timestamp when the summary was last updated';
-- Index: idx_session_summaries_created_at

-- DROP INDEX IF EXISTS public.idx_session_summaries_created_at;

CREATE INDEX IF NOT EXISTS idx_session_summaries_created_at
    ON public.chat_session_summaries USING btree
    (created_at ASC NULLS LAST)
    WITH (fillfactor=100, deduplicate_items=True)
    TABLESPACE pg_default;
-- Index: idx_session_summaries_session_id

-- DROP INDEX IF EXISTS public.idx_session_summaries_session_id;

CREATE INDEX IF NOT EXISTS idx_session_summaries_session_id
    ON public.chat_session_summaries USING btree
    (session_id COLLATE pg_catalog."default" ASC NULLS LAST)
    WITH (fillfactor=100, deduplicate_items=True)
    TABLESPACE pg_default;
