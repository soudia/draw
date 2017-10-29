publish = $(foreach port,$(OPEN_PORTS),--publish $(port))
mount = $(foreach volume,$(MOUNTED_VOLUMES),-v $(volume))

.PHONY: build
build:
	docker build -t $(IMAGE_NAME) .

.PHONY: rebuild
rebuild:
	docker build --no-cache -t $(IMAGE_NAME) .


.PHONY: run
run: build rm
	$(NV_HEADER) nvidia-docker run --detach=true $(publish) $(mount) --name $(CONTAINER_NAME) --workdir=$(WORK_DIR) $(IMAGE_NAME)

.PHONY: exec
exec:
	docker exec --interactive --tty $(CONTAINER_NAME) /bin/bash


.PHONY: stop
stop:
	-docker stop $(CONTAINER_NAME)


.PHONY: rm
rm: stop
	-docker rm $(CONTAINER_NAME)
